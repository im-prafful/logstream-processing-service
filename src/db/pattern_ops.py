from sqlalchemy import text


def save_pattern(engine):
    """
    Fetches one representative log (pattern) for each cluster
    and inserts it into the log_patterns table.
    """

    # Fetch the first log entry for each cluster (the pattern)
    # along with the cluster's total count.
    query = text(
        """
        SELECT 
            concat_ws(' | ', l.source, l.level, l.message, l.parsed_data) AS merged_string,
            l.cluster_id,
            l.app_id, -- Need app_id for the insert
            t.total_count
        FROM logs l
        JOIN (
            SELECT cluster_id, MIN(log_id) AS first_log, COUNT(*) AS total_count
            FROM logs
            GROUP BY cluster_id
            HAVING cluster_id IS NOT NULL -- Only consider logs that have been clustered
        ) t
        ON l.cluster_id = t.cluster_id AND l.log_id = t.first_log;
    """
    )

    # Insert log patterns into the log_patterns table.
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
        result = conn.execute(query)
        # Fetch all rows, the schema is (merged_string, cluster_id, app_id, total_count)
        rows = result.fetchall()

        print(f"Fetched {len(rows)} distinct log patterns.")

        # Step B: Prepare parameters for bulk insertion
        insert_params = [
            {
                "app_id": row[2],  # app_id
                "log_template": row[0],  # merged_string
                "incident_count": row[
                    3
                ],  # total_count (or 1 depending on intent, using total_count here)
                "cluster_id": row[1],  # cluster_id
            }
            for row in rows
        ]

        # Step C: Execute  insertion
        if insert_params:
            conn.execute(insert_pattern_query, insert_params)
            print(f"Inserted/updated {len(insert_params)} log patterns.")
