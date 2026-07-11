import json
import boto3
from sqlalchemy import text

from src.db.cluster_ops import save_cluster_stats, fetch_cluster_history

def publish_anomaly_event(conn, cluster_id, reason):
    try:
        # 1. Fetch context (top 5 logs)
        query = text("SELECT message FROM logs WHERE cluster_id = :cid ORDER BY log_id DESC LIMIT 5")
        rows = conn.execute(query, {"cid": cluster_id}).fetchall()
        logs = [row[0] for row in rows]
        
        # 2. Publish to EventBridge
        client = boto3.client('events', region_name='ap-south-1')
        detail = {
            "cluster_id": cluster_id,
            "reason": reason,
            "sample_logs": logs
        }
        
        response = client.put_events(
            Entries=[
                {
                    'Source': 'com.logstream.processing',
                    'DetailType': 'VolumeAnomalyDetected',
                    'Detail': json.dumps(detail),
                    'EventBusName': 'Logstream-alert-bus'
                }
            ]
        )
        print(f"Published EventBridge anomaly event for Cluster {cluster_id}")
    except Exception as e:
        print(f"Failed to publish EventBridge event for Cluster {cluster_id}: {e}")


def determine_assigned_role(sample_logs: list) -> str:
    """
    Heuristically determines the best engineering team to handle the incident
    based on the structural signatures of the sample logs.
    """
    dev_signatures = ["exception", "error:", "traceback", "nullpointer", "typeerror", "syntax"]
    qa_signatures = ["test failed", "assertion", "timeout during test", "validation failed", "jest", "cypress"]
    
    dev_score = 0
    qa_score = 0
    
    for log in sample_logs:
        log_lower = str(log).lower()
        if any(sig in log_lower for sig in dev_signatures):
            dev_score += 1
        if any(sig in log_lower for sig in qa_signatures):
            qa_score += 1
            
    if dev_score > 0 and dev_score >= qa_score:
        return "dev"
    elif qa_score > 0 and qa_score > dev_score:
        return "qa"
        
    return "sre"
    
def create_incident(engine, cluster_id, reason="Volume Anomaly"):
    check_query = text(
        """
            SELECT 1
            FROM incidents
            WHERE cluster_id = :cid AND status IN ('INPROGRESS', 'NEW')
            LIMIT 1
        """
    )

    update_query = text(
        """
            UPDATE incidents
            SET updated_at = NOW()
            WHERE cluster_id = :cid AND status IN ('INPROGRESS', 'NEW')
        """
    )

    insert_query = text(
        """
            INSERT INTO incidents (cluster_id,status,assigned_role,assigned_to,created_at,updated_at,resolved_at)
            VALUES(:cid,'NEW',:role,null,NOW(),null,null)
        """
    )

    with engine.begin() as conn:
        existing_open = conn.execute(check_query, {"cid": cluster_id}).fetchone()
        if existing_open:
            conn.execute(update_query, {"cid": cluster_id})
            print(
                f"Incident already active for Cluster {cluster_id}; refreshed timestamp [{reason}]"
            )
            return

        # Fetch sample logs for heuristic routing (using 30 logs for high accuracy on the cluster)
        log_query = text("SELECT message FROM logs WHERE cluster_id = :cid ORDER BY log_id DESC LIMIT 30")
        log_rows = conn.execute(log_query, {"cid": cluster_id}).fetchall()
        sample_logs = [row[0] for row in log_rows]
        
        # Determine the best role based on the clustered logs
        assigned_role = determine_assigned_role(sample_logs)

        conn.execute(insert_query, {"cid": cluster_id, "role": assigned_role})
        print(f"New Incident CREATED for Cluster {cluster_id} [{reason}] -> Routed to {assigned_role.upper()}")
        
        # Instantly publish the rich-context event to EventBridge!
        publish_anomaly_event(conn, cluster_id, reason)


def detect_and_create_incidents(engine, start_log_id, end_log_id):
    """
    End-of-batch orchestrator: saves cluster volume stats,
    runs anomaly detection, and creates incidents for flagged clusters.
    """
    from src.ml.volume_analyzer import VolumeAnomalyDetector

    # 1. Count how many logs landed in each cluster during this batch
    count_query = text(
        """
        SELECT cluster_id, COUNT(*) as cnt
        FROM logs
        WHERE cluster_id IS NOT NULL
          AND level IN ('error','warning')
          AND log_id BETWEEN :start_log_id AND :end_log_id
        GROUP BY cluster_id
    """
    )

    try:
        with engine.begin() as conn:
            rows = conn.execute(
                count_query,
                {"start_log_id": start_log_id, "end_log_id": end_log_id},
            ).fetchall()
        batch_stats = {row[0]: row[1] for row in rows}
    except Exception as e:
        print(f"Error counting cluster stats: {e}")
        return

    print(f"Batch cluster counts: {batch_stats}")

    # 2. Save stats to history
    save_cluster_stats(engine, batch_stats)

    # 3. Fetch history window
    history_df = fetch_cluster_history(engine, window_size=5)


    # 4. Load volume model and detect anomalies
    vol_detector = VolumeAnomalyDetector(window_size=5)
    vol_detector.load("scripts/models/production")
    anomalous_clusters = vol_detector.detect_anomalies(history_df)

    # 5. Sanity guard: if anomaly ratio is unreasonably high, skip
    total_evaluated = history_df["cluster_id"].nunique()
    MAX_ANOMALY_RATIO = 0.3
    if total_evaluated > 0 and len(anomalous_clusters) > 0:
        ratio = len(anomalous_clusters) / total_evaluated
        if ratio > MAX_ANOMALY_RATIO:
            print(
                f"⚠️ Anomaly ratio too high: {len(anomalous_clusters)}/{total_evaluated} "
                f"({ratio:.0%}). Likely model miscalibration. Skipping incident creation."
            )
            return

    # 6. Create incidents
    if anomalous_clusters:
        print(f"🚨 Creating incidents for {len(anomalous_clusters)} anomalous clusters.")
        for cid in anomalous_clusters:
            # Only alert if the cluster ACTUALLY appeared in this current batch
            if cid in batch_stats:
                create_incident(engine, cid, reason="Volume Anomaly")
    else:
        print("✅ No volume anomalies detected.")

