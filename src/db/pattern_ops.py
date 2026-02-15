from sqlalchemy import text


def save_pattern(engine):
    """
    Fetches one representative log (pattern) for each cluster
    and inserts it into the log_patterns table.
    Only fetches patterns newer than the last seen timestamp (if one exists).
    """

    try:
        # Check for the latest timestamp in the patterns table
        null_check = text(
            """
                SELECT MAX(last_seen) AS last_timestamp
                FROM patter_table;
            """
        )

        last_time = None
        with engine.begin() as conn:
            result = conn.execute(null_check)
            row = result.fetchone()
            if row:
                last_time = row[0]

        # Build query based on whether we have a previous timestamp
        if last_time:
            query = text(
                """
                SELECT 
                    concat_ws(' | ', l.source, l.level, l.message, l.parsed_data) AS merged_string,
                    l.cluster_id,
                    l.app_id,
                    t.total_count
                FROM logs l
                JOIN (
                    SELECT cluster_id, MIN(log_id) AS first_log, COUNT(*) AS total_count
                    FROM logs
                    GROUP BY cluster_id
                    HAVING cluster_id IS NOT NULL
                ) t
                ON l.cluster_id = t.cluster_id AND l.log_id = t.first_log
                WHERE l.last_seen > :last_time;
                """
            )
            query_params = {"last_time": last_time}
        else:
            query = text(
                """
                SELECT 
                    concat_ws(' | ', l.source, l.level, l.message, l.parsed_data) AS merged_string,
                    l.cluster_id,
                    l.app_id,
                    t.total_count
                FROM logs l
                JOIN (
                    SELECT cluster_id, MIN(log_id) AS first_log, COUNT(*) AS total_count
                    FROM logs
                    GROUP BY cluster_id
                    HAVING cluster_id IS NOT NULL
                ) t
                ON l.cluster_id = t.cluster_id AND l.log_id = t.first_log;
                """
            )
            query_params = {}

        # Insert log patterns into the log_patterns table
        insert_pattern_query = text(
            """
            INSERT INTO log_patterns (
                app_id, log_template, incident_count, cluster_id
            )
            VALUES (
                :app_id, 
                :log_template, 
                :incident_count, 
                :cluster_id
            )
            """
        )

        with engine.begin() as conn:
            # Step A: Fetch all pattern data
            result = conn.execute(query, query_params)
            rows = result.fetchall()

            print(f"Fetched {len(rows)} distinct log patterns.")

            # Step B: Prepare parameters for bulk insertion
            insert_params = [
                {
                    "app_id": row[2],
                    "log_template": row[0],
                    "incident_count": row[3],
                    "cluster_id": row[1],
                }
                for row in rows
            ]

            # Step C: Execute insertion
            if insert_params:
                conn.execute(insert_pattern_query, insert_params)
                print(f"Inserted/updated {len(insert_params)} log patterns.")

    except Exception as e:
        print(f"Error in save_pattern: {e}")
