from sqlalchemy import text

from src.db.cluster_ops import save_cluster_stats, fetch_cluster_history


def create_incident(engine, cluster_id, reason="Volume Anomaly"):

    insert_query = text(
        """
            INSERT INTO incidents (cluster_id,status,assigned_role,assigned_to,created_at,updated_at,resolved_at)
            VALUES(:cid,'NEW','SRE',null,NOW(),null,null)
        """
    )

    with engine.begin() as conn:
        existing_open = conn.execute(check_query, {"cid": cluster_id}).fetchone()
        if existing_open:
            conn.execute(update_query, {"cid": cluster_id})
            print(f"Incident already OPEN for Cluster {cluster_id}; refreshed timestamp [{reason}]")
            return

            conn.execute(insert_query, {"cid": cluster_id})
            print(f"New Incident CREATED for Cluster {cluster_id} [{reason}]")


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

    # 2. Save stats to history
    save_cluster_stats(engine, batch_stats)

    # 3. Fetch history window
    history_df = fetch_cluster_history(engine, window_size=5)

    # 4. Load volume model and detect anomalies
    vol_detector = VolumeAnomalyDetector(window_size=5)
    vol_detector.load("scripts/models/production")
    anomalous_clusters = vol_detector.detect_anomalies(history_df)

    # 5. Create incidents
    if anomalous_clusters:
        print(f"Detected {len(anomalous_clusters)} anomalous clusters!")
        for cid in anomalous_clusters:
            create_incident(engine, cid, reason="Volume Anomaly")
    else:
        print("No volume anomalies detected.")
