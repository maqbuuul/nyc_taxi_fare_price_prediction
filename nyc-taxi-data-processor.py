import os
import duckdb
import zipfile
import shutil
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Define paths
DATA_FOLDER = "./data/raw/"
DB_FOLDER = "./db/"
DB_PATH = os.path.join(DB_FOLDER, "nyc_taxi.duckdb")
TEMP_FOLDER = "./temp_shapefiles"
GEOJSON_OUTPUT_PATH = os.path.join(DB_FOLDER, "taxi_zones.geojson")
LOOKUP_CSV_OUTPUT = os.path.join(DATA_FOLDER, "taxi_zone_lookup.csv")
ZIP_FILE = os.path.join(DATA_FOLDER, "taxi_zones.zip")

# Target year range for taxi data
TARGET_START_YEAR = 2022
TARGET_END_YEAR = 2024

# Create output directories if they don't exist
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)


# Function to process taxi zone shapefile
def process_taxi_zones():
    """Process taxi zone data and create GeoJSON and lookup CSV"""
    print("\n--- Processing Taxi Zone Data ---")

    try:
        # Check if GeoJSON already exists
        if os.path.exists(GEOJSON_OUTPUT_PATH):
            print(f"GeoJSON file already exists at {GEOJSON_OUTPUT_PATH}")
            return

        # Check if taxi_zones.zip exists
        if os.path.exists(ZIP_FILE):
            print(f"Found taxi zones zip file: {ZIP_FILE}")
            # Extract the zip file
            with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
                zip_ref.extractall(TEMP_FOLDER)

            # Check for shapefile
            shapefiles = []
            for root, dirs, files in os.walk(TEMP_FOLDER):
                shapefiles.extend(
                    [os.path.join(root, f) for f in files if f.endswith(".shp")]
                )

            if shapefiles:
                print(f"Found shapefile: {shapefiles[0]}")
                # Try to use geopandas if available
                try:
                    import geopandas as gpd

                    gdf = gpd.read_file(shapefiles[0])
                    print(f"Converting shapefile to GeoJSON: {GEOJSON_OUTPUT_PATH}")
                    gdf.to_file(GEOJSON_OUTPUT_PATH, driver="GeoJSON")

                    # Create lookup CSV if needed
                    columns = gdf.columns.tolist()
                    print(f"Shapefile columns: {columns}")

                    # Try to identify key columns
                    zone_col = None
                    borough_col = None
                    id_col = None

                    for col in columns:
                        if col.lower() == "zone" or "zone" in col.lower():
                            zone_col = col
                        elif col.lower() == "borough" or "borough" in col.lower():
                            borough_col = col
                        elif "id" in col.lower() or "location" in col.lower():
                            id_col = col

                    if zone_col:
                        print(
                            f"Creating lookup CSV using column '{zone_col}' for zones"
                        )
                        # Create lookup DataFrame
                        lookup_df = pd.DataFrame()

                        if id_col:
                            lookup_df["LocationID"] = gdf[id_col]
                        else:
                            lookup_df["LocationID"] = range(1, len(gdf) + 1)

                        if borough_col:
                            lookup_df["Borough"] = gdf[borough_col]
                        else:
                            lookup_df["Borough"] = "Unknown"

                        lookup_df["Zone"] = gdf[zone_col]

                        # Save lookup CSV
                        print(f"Saving lookup CSV to {LOOKUP_CSV_OUTPUT}")
                        lookup_df.to_csv(LOOKUP_CSV_OUTPUT, index=False)
                    else:
                        print("Couldn't identify zone column in shapefile")
                        create_dummy_taxi_zone_data()

                except ImportError:
                    print("GeoPandas not available, creating dummy data instead")
                    create_dummy_taxi_zone_data()
                except Exception as e:
                    print(f"Error processing shapefile: {str(e)}")
                    create_dummy_taxi_zone_data()
            else:
                print("No shapefile found in the zip file")
                create_dummy_taxi_zone_data()
        else:
            print(f"Taxi zones zip file not found at {ZIP_FILE}")
            # Check for alternative zip files
            zip_alternatives = [
                f
                for f in os.listdir(DATA_FOLDER)
                if f.endswith(".zip") and "zone" in f.lower()
            ]
            if zip_alternatives:
                print(f"Found alternative zip file: {zip_alternatives[0]}")
                alt_zip_path = os.path.join(DATA_FOLDER, zip_alternatives[0])
                with zipfile.ZipFile(alt_zip_path, "r") as zip_ref:
                    zip_ref.extractall(TEMP_FOLDER)
                # Recursively call this function to process the extracted files
                process_taxi_zones()
            else:
                # Check for map images
                map_files = [
                    f
                    for f in os.listdir(DATA_FOLDER)
                    if f.startswith("taxi_zone_map_")
                    and f.endswith((".jpg", ".png", ".jpeg"))
                ]
                if map_files:
                    print(f"Found map images: {len(map_files)} files")
                    # We can't directly convert images to GeoJSON, so create dummy data
                    create_dummy_taxi_zone_data()
                else:
                    print("No taxi zone data sources found")
                    create_dummy_taxi_zone_data()

    finally:
        # Clean up temporary files
        if os.path.exists(TEMP_FOLDER):
            print("Cleaning up temporary files...")
            shutil.rmtree(TEMP_FOLDER, ignore_errors=True)


def create_dummy_taxi_zone_data():
    """Create dummy taxi zone data when real data is unavailable"""
    print("Creating dummy taxi zone data...")

    # Create a basic GeoJSON with NYC boroughs and zones
    boroughs = [
        {
            "name": "Manhattan",
            "id": 1,
            "zones": [
                "Midtown",
                "Downtown",
                "Upper East Side",
                "Upper West Side",
                "Harlem",
            ],
        },
        {
            "name": "Brooklyn",
            "id": 2,
            "zones": [
                "Downtown Brooklyn",
                "Williamsburg",
                "Park Slope",
                "Bay Ridge",
                "Coney Island",
            ],
        },
        {
            "name": "Queens",
            "id": 3,
            "zones": [
                "Astoria",
                "Long Island City",
                "Flushing",
                "Jamaica",
                "JFK Airport",
            ],
        },
        {
            "name": "Bronx",
            "id": 4,
            "zones": [
                "South Bronx",
                "Fordham",
                "Riverdale",
                "Pelham Bay",
                "Mott Haven",
            ],
        },
        {
            "name": "Staten Island",
            "id": 5,
            "zones": [
                "St. George",
                "Stapleton",
                "Great Kills",
                "Tottenville",
                "New Dorp",
            ],
        },
    ]

    features = []
    location_id = 1
    lookup_data = []

    for borough in boroughs:
        # For each borough, create a simple rectangular area
        lon_start = -74.05 + (borough["id"] - 1) * 0.05
        lat_start = 40.60 + (borough["id"] - 1) * 0.04

        # Create zones within borough
        for i, zone in enumerate(borough["zones"]):
            # Create a small zone box within the borough
            zone_lon = lon_start + 0.01 * (i % 3)
            zone_lat = lat_start + 0.01 * (i // 3)

            zone_coords = [
                [zone_lon, zone_lat],
                [zone_lon + 0.01, zone_lat],
                [zone_lon + 0.01, zone_lat + 0.01],
                [zone_lon, zone_lat + 0.01],
                [zone_lon, zone_lat],  # Close polygon
            ]

            feature = {
                "type": "Feature",
                "properties": {
                    "LocationID": location_id,
                    "Borough": borough["name"],
                    "Zone": zone,
                },
                "geometry": {"type": "Polygon", "coordinates": [zone_coords]},
            }
            features.append(feature)

            # Add to lookup data
            lookup_data.append(
                {"LocationID": location_id, "Borough": borough["name"], "Zone": zone}
            )

            location_id += 1

    # Create GeoJSON structure
    geojson = {"type": "FeatureCollection", "features": features}

    # Save GeoJSON
    print(f"Saving dummy GeoJSON to {GEOJSON_OUTPUT_PATH}")
    with open(GEOJSON_OUTPUT_PATH, "w") as f:
        json.dump(geojson, f)

    # Save lookup CSV
    lookup_df = pd.DataFrame(lookup_data)
    print(f"Saving dummy taxi zone lookup table to {LOOKUP_CSV_OUTPUT}")
    lookup_df.to_csv(LOOKUP_CSV_OUTPUT, index=False)

    print("✅ Created dummy GeoJSON and lookup CSV with basic NYC borough information.")


def process_weather_data():
    """Process NYC weather data CSV files into DuckDB database"""
    print("\n--- Processing Weather Data ---")

    # Connect to DuckDB database
    print(f"Connecting to database at {DB_PATH}")
    con = duckdb.connect(DB_PATH)

    # Drop existing weather table
    con.execute("DROP TABLE IF EXISTS nyc_weather")

    # Find all possible weather files
    weather_files = [
        f
        for f in os.listdir(DATA_FOLDER)
        if ("york" in f.lower() or "weather" in f.lower() or "ny" in f.lower())
        and (f.endswith(".csv") or f.endswith(".xlsx") or f.endswith(".xls"))
    ]

    if not weather_files:
        print(
            "No weather data files found. Please add weather CSV files to the data/raw directory."
        )
        return

    print(f"Found {len(weather_files)} weather data files: {', '.join(weather_files)}")

    # Process each weather file
    first_file = True
    for weather_file in weather_files:
        weather_path = os.path.join(DATA_FOLDER, weather_file)
        file_ext = os.path.splitext(weather_file)[1].lower()

        try:
            print(f"Processing weather data file: {weather_file}")

            if file_ext == ".csv":
                # First, examine the CSV to understand its structure
                try:
                    # Read a few rows to check column names
                    preview_df = pd.read_csv(weather_path, nrows=5)
                    print(f"Preview of columns: {preview_df.columns.tolist()}")

                    # Check if the CSV has date/datetime columns
                    date_columns = [
                        col
                        for col in preview_df.columns
                        if "date" in col.lower() or "time" in col.lower()
                    ]
                    print(f"Detected date columns: {date_columns}")

                    if first_file:
                        # Create the table
                        con.execute(
                            f"CREATE TABLE nyc_weather AS SELECT * FROM read_csv_auto('{weather_path}')"
                        )
                        first_file = False
                    else:
                        # Try to insert the data
                        con.execute(
                            f"INSERT INTO nyc_weather SELECT * FROM read_csv_auto('{weather_path}')"
                        )

                    print(f"Successfully loaded weather data from {weather_file}")
                except Exception as e:
                    print(f"Error examining CSV structure: {str(e)}")

            elif file_ext in [".xlsx", ".xls"]:
                # Handle Excel files
                try:
                    if first_file:
                        con.execute(
                            f"CREATE TABLE nyc_weather AS SELECT * FROM read_excel('{weather_path}')"
                        )
                        first_file = False
                    else:
                        con.execute(
                            f"INSERT INTO nyc_weather SELECT * FROM read_excel('{weather_path}')"
                        )

                    print(f"Successfully loaded weather data from {weather_file}")
                except Exception as e:
                    print(f"Error loading Excel file: {str(e)}")

        except Exception as e:
            print(f"Error processing weather file {weather_file}: {str(e)}")

    if first_file:
        # If we couldn't load any files
        print("Failed to load any weather data files.")
        return

    # Check what we've loaded
    row_count = con.execute("SELECT COUNT(*) FROM nyc_weather").fetchone()[0]
    print(f"Loaded {row_count} weather data records into nyc_weather table")

    # Get column names
    columns = con.execute("PRAGMA table_info(nyc_weather)").fetchdf()
    print(f"Weather table columns: {', '.join(columns['name'])}")

    # Try to identify and standardize date columns
    date_cols = [
        col for col in columns["name"] if "date" in col.lower() or "time" in col.lower()
    ]

    # Filter weather data to target years
    if date_cols:
        primary_date_col = date_cols[0]
        print(f"Primary date column identified: {primary_date_col}")

        # Get date range before filtering
        date_range_before = con.execute(
            f'SELECT MIN("{primary_date_col}"), MAX("{primary_date_col}") FROM nyc_weather'
        ).fetchone()
        print(
            f"Weather data date range before filtering: {date_range_before[0]} to {date_range_before[1]}"
        )

        # Filter weather data to target years
        try:
            con.execute(f'''
                DELETE FROM nyc_weather 
                WHERE EXTRACT(YEAR FROM "{primary_date_col}") < {TARGET_START_YEAR} 
                OR EXTRACT(YEAR FROM "{primary_date_col}") > {TARGET_END_YEAR}
            ''')

            # Get new row count and date range
            filtered_count = con.execute("SELECT COUNT(*) FROM nyc_weather").fetchone()[
                0
            ]
            date_range_after = con.execute(
                f'SELECT MIN("{primary_date_col}"), MAX("{primary_date_col}") FROM nyc_weather'
            ).fetchone()

            print(f"Filtered weather data to {TARGET_START_YEAR}-{TARGET_END_YEAR}")
            print(f"Remaining weather records: {filtered_count}")
            print(
                f"Weather data date range after filtering: {date_range_after[0]} to {date_range_after[1]}"
            )
        except Exception as e:
            print(f"Error filtering weather data: {str(e)}")

    # Create index on date column if available
    if date_cols:
        try:
            con.execute(
                f'CREATE INDEX IF NOT EXISTS weather_date_idx ON nyc_weather ("{date_cols[0]}")'
            )
            print(f"Created index on nyc_weather.{date_cols[0]}")
        except Exception as e:
            print(f"Error creating weather date index: {str(e)}")

    print("✅ Weather data processing complete!")
    con.close()


# Function to determine the date column for a table
def get_date_column_for_table(con, table_name):
    """Determine the date column name for a table"""
    columns = con.execute(f"PRAGMA table_info({table_name})").fetchdf()["name"].tolist()

    # Known date column patterns for different taxi types
    date_column_patterns = {
        "yellow_taxi_trips": ["tpep_pickup_datetime", "pickup_datetime"],
        "green_taxi_trips": ["lpep_pickup_datetime", "pickup_datetime"],
        "fhv_trips": ["pickup_datetime", "pickup_date"],
        "hvfhv_trips": ["pickup_datetime", "request_datetime"],
    }

    # Check if table has known date columns
    if table_name in date_column_patterns:
        for col in date_column_patterns[table_name]:
            if col in columns:
                return col

    # Generic search for date columns
    for col in columns:
        if (
            "datetime" in col.lower()
            or "pickup" in col.lower()
            or "date" in col.lower()
        ):
            return col

    # Default fallback
    return None


# Function to validate a parquet file's date range before importing
def validate_parquet_date_range(file_path, date_column=None):
    """Check if a parquet file contains data within our target years"""
    try:
        # Use pandas to read metadata and sample data
        import pyarrow.parquet as pq

        # Read metadata to get schema
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema.to_arrow_schema()

        # If no date column specified, try to find one
        if not date_column:
            schema_fields = [field.name for field in schema]
            for field in schema_fields:
                if (
                    "pickup" in field.lower()
                    or "datetime" in field.lower()
                    or "date" in field.lower()
                ):
                    date_column = field
                    break

        if not date_column:
            # Cannot validate without date column
            print(f"  Warning: Could not identify date column in {file_path}")
            return True  # Assume valid

        # Read a sample of rows to check date range
        sample_df = pd.read_parquet(file_path, columns=[date_column])

        # Handle empty dataframe
        if sample_df.empty:
            print(f"  Warning: No data in {file_path}")
            return False

        # Check date range
        min_year = pd.to_datetime(sample_df[date_column].min()).year
        max_year = pd.to_datetime(sample_df[date_column].max()).year

        print(f"  File date range: {min_year} to {max_year}")

        # Only import if file contains data within our target years
        if max_year < TARGET_START_YEAR or min_year > TARGET_END_YEAR:
            print(
                f"  Skipping file with dates outside target range: {min_year}-{max_year}"
            )
            return False

        return True
    except Exception as e:
        print(f"  Error validating parquet file {file_path}: {str(e)}")
        return True  # Import anyway if validation fails


# Function to process parquet files into DuckDB
def process_parquet_files():
    """Process parquet files into DuckDB database"""
    print("\n--- Processing Parquet Files into DuckDB ---")

    # Connect to DuckDB database
    print(f"Connecting to database at {DB_PATH}")
    con = duckdb.connect(DB_PATH)

    # Define trip types and associated patterns
    trip_types = {
        "yellow_taxi_trips": "yellow_tripdata",
        "green_taxi_trips": "green_tripdata",
        "fhv_trips": "fhv_tripdata",
        "hvfhv_trips": "fhvhv_tripdata",
    }

    # Process each trip type
    for table_name, pattern in trip_types.items():
        print(f"Processing {table_name}...")

        # Drop and create an empty table
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        first_file = True

        # Get all matching files and sort them
        matching_files = [
            f
            for f in os.listdir(DATA_FOLDER)
            if f.endswith(".parquet") and pattern in f
        ]
        matching_files.sort()

        if not matching_files:
            print(f"No files found for {table_name}")
            continue

        # Process each file
        table_date_column = None
        for filename in matching_files:
            file_path = os.path.join(DATA_FOLDER, filename)
            print(f"Loading {file_path}...")

            # Validate if file contains data within our target years
            if not validate_parquet_date_range(file_path, table_date_column):
                continue

            try:
                if first_file:
                    # Create table directly from first file
                    con.execute(
                        f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')"
                    )
                    first_file = False

                    # Determine date column for future filtering
                    table_date_column = get_date_column_for_table(con, table_name)
                    if table_date_column:
                        print(
                            f"  Identified date column for {table_name}: {table_date_column}"
                        )
                    else:
                        print(
                            f"  Warning: Could not identify date column for {table_name}"
                        )
                else:
                    # Insert into table for subsequent files
                    con.execute(
                        f"INSERT INTO {table_name} SELECT * FROM read_parquet('{file_path}')"
                    )
                print(f"  Successfully loaded {filename}")
            except Exception as e:
                print(f"  Error loading {filename}: {str(e)}")

        # If we successfully loaded data, filter by date range
        if not first_file and table_date_column:
            try:
                # Get row count before filtering
                count_before = con.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]
                print(f"Loaded {count_before} rows into {table_name}")

                # Get date range before filtering
                date_range_before = con.execute(
                    f'SELECT MIN("{table_date_column}"), MAX("{table_date_column}") FROM {table_name}'
                ).fetchone()
                print(
                    f"Date range before filtering: {date_range_before[0]} to {date_range_before[1]}"
                )

                # Filter to target years
                con.execute(f'''
                    DELETE FROM {table_name} 
                    WHERE EXTRACT(YEAR FROM "{table_date_column}") < {TARGET_START_YEAR} 
                    OR EXTRACT(YEAR FROM "{table_date_column}") > {TARGET_END_YEAR}
                ''')

                # Get count and date range after filtering
                count_after = con.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]
                date_range_after = con.execute(
                    f'SELECT MIN("{table_date_column}"), MAX("{table_date_column}") FROM {table_name}'
                ).fetchone()

                print(f"Filtered {table_name} to {TARGET_START_YEAR}-{TARGET_END_YEAR}")
                print(
                    f"Removed {count_before - count_after} rows ({(count_before - count_after) / count_before * 100:.2f}%)"
                )
                print(f"Remaining rows: {count_after}")
                print(
                    f"Date range after filtering: {date_range_after[0]} to {date_range_after[1]}"
                )

                # Analyze date distribution with monthly aggregation
                print(f"Analyzing monthly trip counts for {table_name}:")
                monthly_counts = con.execute(f'''
                    SELECT 
                        DATE_TRUNC('month', "{table_date_column}") as month,
                        EXTRACT(YEAR FROM "{table_date_column}") as year,
                        EXTRACT(MONTH FROM "{table_date_column}") as month_num,
                        COUNT(*) as trip_count
                    FROM {table_name}
                    GROUP BY 
                        DATE_TRUNC('month', "{table_date_column}"),
                        EXTRACT(YEAR FROM "{table_date_column}"),
                        EXTRACT(MONTH FROM "{table_date_column}")
                    ORDER BY year, month_num
                ''').fetchdf()

                # Print summary of monthly counts
                print(monthly_counts)

            except Exception as e:
                print(f"Error filtering {table_name} by date: {str(e)}")

    # Load the taxi zone lookup table
    if os.path.exists(LOOKUP_CSV_OUTPUT):
        try:
            print(f"Loading taxi zone lookup table from {LOOKUP_CSV_OUTPUT}")
            con.execute("DROP TABLE IF EXISTS taxi_zone_lookup")
            con.execute(
                f"CREATE TABLE taxi_zone_lookup AS SELECT * FROM read_csv_auto('{LOOKUP_CSV_OUTPUT}')"
            )
            print("Successfully loaded taxi zone lookup table")
        except Exception as e:
            print(f"Error loading taxi zone lookup table: {str(e)}")
    else:
        print("Taxi zone lookup table not found")

    # Create views
    print("Creating views...")
    try:
        # Check if both tables exist before creating the view
        yellow_exists = (
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'yellow_taxi_trips'"
            ).fetchone()[0]
            > 0
        )
        green_exists = (
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'green_taxi_trips'"
            ).fetchone()[0]
            > 0
        )

        if yellow_exists and green_exists:
            # Get column names to ensure compatibility
            yellow_cols = (
                con.execute("PRAGMA table_info(yellow_taxi_trips)")
                .fetchdf()["name"]
                .tolist()
            )
            green_cols = (
                con.execute("PRAGMA table_info(green_taxi_trips)")
                .fetchdf()["name"]
                .tolist()
            )

            # Find common columns
            common_cols = set(yellow_cols).intersection(set(green_cols))
            if common_cols:
                common_cols_str = ", ".join([f'"{col}"' for col in common_cols])
                con.execute(f"""
                    CREATE OR REPLACE VIEW all_taxi_trips AS 
                    SELECT {common_cols_str} FROM yellow_taxi_trips 
                    UNION ALL 
                    SELECT {common_cols_str} FROM green_taxi_trips
                """)
                print("Successfully created all_taxi_trips view")
            else:
                print("No common columns found between yellow and green taxi tables")
        else:
            print(
                "Yellow or green taxi table doesn't exist, skipping all_taxi_trips view"
            )

        # Check if both FHV tables exist
        fhv_exists = (
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'fhv_trips'"
            ).fetchone()[0]
            > 0
        )
        hvfhv_exists = (
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'hvfhv_trips'"
            ).fetchone()[0]
            > 0
        )

        if fhv_exists and hvfhv_exists:
            # Get column names to ensure compatibility
            fhv_cols = (
                con.execute("PRAGMA table_info(fhv_trips)").fetchdf()["name"].tolist()
            )
            hvfhv_cols = (
                con.execute("PRAGMA table_info(hvfhv_trips)").fetchdf()["name"].tolist()
            )

            # Find common columns
            common_cols = set(fhv_cols).intersection(set(hvfhv_cols))
            if common_cols:
                common_cols_str = ", ".join([f'"{col}"' for col in common_cols])
                con.execute(f"""
                    CREATE OR REPLACE VIEW all_fhv_trips AS 
                    SELECT {common_cols_str} FROM fhv_trips 
                    UNION ALL 
                    SELECT {common_cols_str} FROM hvfhv_trips
                """)
                print("Successfully created all_fhv_trips view")
            else:
                print("No common columns found between FHV and HVFHV tables")
        else:
            print("FHV or HVFHV table doesn't exist, skipping all_fhv_trips view")

    except Exception as e:
        print(f"Error creating views: {str(e)}")

    # Create indexes for better performance
    print("Creating indexes...")
    try:
        # Check if yellow_taxi_trips table exists and has the necessary columns
        yellow_exists = (
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'yellow_taxi_trips'"
            ).fetchone()[0]
            > 0
        )
        if yellow_exists:
            yellow_cols = (
                con.execute("PRAGMA table_info(yellow_taxi_trips)")
                .fetchdf()["name"]
                .tolist()
            )
            if "tpep_pickup_datetime" in yellow_cols:
                con.execute(
                    "CREATE INDEX IF NOT EXISTS yellow_pickup_idx ON yellow_taxi_trips (tpep_pickup_datetime)"
                )
                print("Created index on yellow_taxi_trips.tpep_pickup_datetime")
            if "tpep_dropoff_datetime" in yellow_cols:
                con.execute(
                    "CREATE INDEX IF NOT EXISTS yellow_dropoff_idx ON yellow_taxi_trips (tpep_dropoff_datetime)"
                )
                print("Created index on yellow_taxi_trips.tpep_dropoff_datetime")

        # Check if green_taxi_trips table exists and has the necessary columns
        green_exists = (
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'green_taxi_trips'"
            ).fetchone()[0]
            > 0
        )
        if green_exists:
            green_cols = (
                con.execute("PRAGMA table_info(green_taxi_trips)")
                .fetchdf()["name"]
                .tolist()
            )
            if "lpep_pickup_datetime" in green_cols:
                con.execute(
                    "CREATE INDEX IF NOT EXISTS green_pickup_idx ON green_taxi_trips (lpep_pickup_datetime)"
                )
                print("Created index on green_taxi_trips.lpep_pickup_datetime")
            if "lpep_dropoff_datetime" in green_cols:
                con.execute(
                    "CREATE INDEX IF NOT EXISTS green_dropoff_idx ON green_taxi_trips (lpep_dropoff_datetime)"
                )
                print("Created index on green_taxi_trips.lpep_dropoff_datetime")

    except Exception as e:
        print(f"Error creating indexes: {str(e)}")

    # Analyze data quality for all tables
    print("\n--- Data Quality Analysis ---")
    # Get all tables
    tables = (
        con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        )
        .fetchdf()["table_name"]
        .tolist()
    )

    for table in tables:
        try:
            # Skip system tables
            if table.startswith("sqlite_"):
                continue

            print(f"\nAnalyzing table: {table}")

            # Get row count
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"Total rows: {count:,}")

            # Get column names
            columns = con.execute(f"PRAGMA table_info({table})").fetchdf()
            print(f"Columns: {', '.join(columns['name'])}")

            # Identify date/time columns
            date_cols = [
                col
                for col in columns["name"]
                if "date" in col.lower() or "time" in col.lower()
            ]

            # Check date ranges if date columns exist
            if date_cols:
                for date_col in date_cols:
                    try:
                        date_range = con.execute(
                            f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM {table}'
                        ).fetchone()
                        print(
                            f"Date range for column '{date_col}': {date_range[0]} to {date_range[1]}"
                        )

                        # Check for outlier years
                        year_counts = con.execute(f'''
                            SELECT 
                                EXTRACT(YEAR FROM "{date_col}") as year, 
                                COUNT(*) as count
                            FROM {table}
                            GROUP BY EXTRACT(YEAR FROM "{date_col}")
                            ORDER BY year
                        ''').fetchdf()

                        print("Year distribution:")
                        print(year_counts)

                        # Check for records outside target range
                        outliers = con.execute(f'''
                            SELECT COUNT(*) 
                            FROM {table} 
                            WHERE EXTRACT(YEAR FROM "{date_col}") < {TARGET_START_YEAR} 
                            OR EXTRACT(YEAR FROM "{date_col}") > {TARGET_END_YEAR}
                        ''').fetchone()[0]

                        if outliers > 0:
                            print(
                                f"WARNING: Found {outliers} records outside target years {TARGET_START_YEAR}-{TARGET_END_YEAR}"
                            )
                            print(f"Removing outlier records...")

                            # Remove outliers
                            con.execute(f'''
                                DELETE FROM {table}
                                WHERE EXTRACT(YEAR FROM "{date_col}") < {TARGET_START_YEAR} 
                                OR EXTRACT(YEAR FROM "{date_col}") > {TARGET_END_YEAR}
                            ''')

                            # Check new count
                            new_count = con.execute(
                                f"SELECT COUNT(*) FROM {table}"
                            ).fetchone()[0]
                            print(f"Removed {count - new_count} outlier records")
                            print(f"New row count: {new_count:,}")

                            # Verify date range after cleaning
                            clean_date_range = con.execute(
                                f'SELECT MIN("{date_col}"), MAX("{date_col}") FROM {table}'
                            ).fetchone()
                            print(
                                f"Clean date range: {clean_date_range[0]} to {clean_date_range[1]}"
                            )
                    except Exception as e:
                        print(f"Error analyzing date column '{date_col}': {str(e)}")

            # Check for null values in key columns
            for col in columns["name"]:
                try:
                    null_count = con.execute(
                        f'SELECT COUNT(*) FROM {table} WHERE "{col}" IS NULL'
                    ).fetchone()[0]
                    if null_count > 0:
                        null_percent = (null_count / count) * 100
                        print(
                            f"Column '{col}' has {null_count:,} NULL values ({null_percent:.2f}%)"
                        )
                except Exception as e:
                    print(f"Error checking nulls in column '{col}': {str(e)}")

        except Exception as e:
            print(f"Error analyzing table {table}: {str(e)}")

    # Create verification query to check date ranges
    print("\n--- Final Verification ---")
    try:
        # For yellow taxi
        if "yellow_taxi_trips" in tables:
            date_col = get_date_column_for_table(con, "yellow_taxi_trips")
            if date_col:
                monthly_summary = con.execute(f'''
                    SELECT 
                        DATE_TRUNC('month', "{date_col}") as month,
                        EXTRACT(YEAR FROM "{date_col}") as year,
                        EXTRACT(MONTH FROM "{date_col}") as month_num,
                        COUNT(*) as trip_count,
                        SUM(fare_amount) as total_fare,
                        AVG(fare_amount) as avg_fare,
                        SUM(tip_amount) as total_tip,
                        AVG(tip_amount) as avg_tip
                    FROM yellow_taxi_trips
                    GROUP BY 
                        DATE_TRUNC('month', "{date_col}"),
                        EXTRACT(YEAR FROM "{date_col}"),
                        EXTRACT(MONTH FROM "{date_col}")
                    ORDER BY year, month_num
                ''').fetchdf()

                print("\nYellow Taxi Monthly Summary:")
                print(monthly_summary)

        # For green taxi
        if "green_taxi_trips" in tables:
            date_col = get_date_column_for_table(con, "green_taxi_trips")
            if date_col:
                monthly_summary = con.execute(f'''
                    SELECT 
                        DATE_TRUNC('month', "{date_col}") as month,
                        EXTRACT(YEAR FROM "{date_col}") as year,
                        EXTRACT(MONTH FROM "{date_col}") as month_num,
                        COUNT(*) as trip_count
                    FROM green_taxi_trips
                    GROUP BY 
                        DATE_TRUNC('month', "{date_col}"),
                        EXTRACT(YEAR FROM "{date_col}"),
                        EXTRACT(MONTH FROM "{date_col}")
                    ORDER BY year, month_num
                ''').fetchdf()

                print("\nGreen Taxi Monthly Summary:")
                print(monthly_summary)

    except Exception as e:
        print(f"Error in final verification: {str(e)}")

    print("✅ Database processing complete!")
    con.close()


# Main execution
if __name__ == "__main__":
    print("Starting NYC Taxi Data Processing")
    print(f"Target date range: {TARGET_START_YEAR}-{TARGET_END_YEAR}")

    # First process the taxi zone data to ensure we have the lookup file
    process_taxi_zones()

    # Process the weather data
    process_weather_data()

    # Then process the parquet files
    process_parquet_files()

    print("\n✅ All processing complete! Database is ready.")
    print(
        f"Database filtered to contain only data from {TARGET_START_YEAR} to {TARGET_END_YEAR}"
    )
