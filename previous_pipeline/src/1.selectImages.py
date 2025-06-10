
import csv
import json
from datetime import datetime
from utility import *

def write_csv(metadata_path, csv_path):

    """
    Not used
    """

    #print(metadata_path)
    with open(csv_path,'w+') as f:

        writer = csv.writer(f)
        header = ['name', 'cloudCover', 'timestamp','nodata_percentage','orbit']
        writer.writerow(header)

        for root, _, files in os.walk(metadata_path):
            for file in files:
                #print(file)
                if file.endswith('.json'):
                    l2a_tileinfo_path = os.path.join(root,file)
                    xml_tileinfo_path = l2a_tileinfo_path.replace('parsed','original')
                    xml_tileinfo_path = xml_tileinfo_path.replace('tileInfo.json','metadata.xml')

                    try:
                        with open(l2a_tileinfo_path) as json_data, open(xml_tileinfo_path) as xml_data:
                            
                            #load json
                            json_tileinfo = json.load(json_data)

                            # Get Filename
                            filename = json_tileinfo['productName']

                            # Get cloud coverage
                            cloud_line = ''
                            xml_lines = xml_data.readlines()
                            for line in xml_lines:
                                if '<CLOUDY_PIXEL_PERCENTAGE>' in line:
                                    cloud_line = line
                                if '<SENSING_TIME metadataLevel="Standard">' in line:
                                    sensing_time_line = line
                                if '<NODATA_PIXEL_PERCENTAGE>' in line:
                                    nodata_line = line

                            cloud_coverage = float(cloud_line.replace('<CLOUDY_PIXEL_PERCENTAGE>','').replace('</CLOUDY_PIXEL_PERCENTAGE>',''))
                            sensing_time = sensing_time_line.replace('<SENSING_TIME metadataLevel="Standard">','') \
                                                            .replace('</SENSING_TIME>','').replace('"','').replace('    ','').replace('\n','')
                            nodata = nodata_line.replace('<NODATA_PIXEL_PERCENTAGE>','').replace('</NODATA_PIXEL_PERCENTAGE>','') \
                                                .replace('</SENSING_TIME>','').replace('"','').replace('  ','').replace('\n','')
                            orbit = filename.split('_')[4]

                            data = [filename,str(cloud_coverage),sensing_time,str(nodata),orbit]

                            if cloud_coverage <= configuration["cloud_threshold"]:
                                writer.writerow(data)

                    except Exception as e:
                        print ('Exception: ', e)
                        print ('***********************************Skipped******************************')
                        continue

def filter_images_by_date_range(df, start_date, end_date, year):
    filtered_images = []
    for index in range(len(df)):

        img_row = df.iloc[[index]]
        img_date = datetime.strptime(img_row.timestamp_.tolist()[0], "%Y-%m-%d")

        if start_date <= img_date <= end_date:
            filtered_images.append(img_row)
            
    return filtered_images


def reduce_number_of_images(configuration):
    """
    Reduces the number of satellite images by filtering and selecting based on specific criteria.

    Parameters:
    - csv_path: Path to the CSV file containing metadata about the satellite images.

    Steps:
    1. Load and preprocess the CSV data, extracting relevant timestamp and orbit information.
    2. For each bi-monthly period:
       a. Filter images within the bi-monthly date range.
       b. If the number of images exceeds the threshold, select the best images based on cloud cover and orbit representation.
       c. If necessary, fill in missing images from adjacent months.
       d. Ensure all orbits are represented, adding images with the lowest nodata_percentage if needed.
    3. Return the final set of selected images.
    """
    
    # Load the CSV data
    df = pd.read_csv(configuration["csv_data_path"])
    df = df.dropna()  # Drop rows with any missing values
    df['nameCopy'] = df.name.str[4:]  # Create a copy of the 'name' column without the first 4 characters
    df['timestamp_'] = pd.to_datetime(df['timestamp'])  # Convert the timestamp column to datetime

    # Extract year, month and day into separate columns
    df['year'] = df['timestamp_'].dt.year
    df['month'] = df['timestamp_'].dt.month
    df['day'] = df['timestamp_'].dt.day

    # Update the timestamp column to only include year, month and day
    df['timestamp_'] = df['timestamp_'].dt.strftime('%Y-%m-%d')

    all_orbits = df['orbit'].unique()  # Get unique orbits
    year = '2019'  # Set the year of interest
    final_images = []  # Initialize list to store final selected images

    # Print initial information about the dataset
    print(f"\nREADING INPUT DATA\n")
    print(f" - From {configuration['csv_data_path']}")
    print(f" - Initial dataframe size: {len(df)}")
    print(f" - Identified {len(all_orbits)} unique orbits for year {year}")
    print(f" - Max. Timestamp: {df['timestamp'].max()}")
    print(f" - Min. Timestamp: {df['timestamp'].min()}")
    print(f" - Max. cloudCover: {df['cloudCover'].max()}")
    print(f" - Min. cloudCover: {df['cloudCover'].min()}")
    print(f" - Max. nodata_percentage: {df['nodata_percentage'].max()}")
    print(f" - Min. nodata_percentage: {df['nodata_percentage'].min()}")
    print(f"\n")

    # Loop through each bi-monthly period (each month_idx corresponds to two months)
    for month_idx in range(12):
        # Define the start and end dates for the bi-monthly range
        bi_monthly_range = configuration["bi_monthly_ranges"][month_idx]
        start_date = datetime.strptime(year + bi_monthly_range[0], "%Y%m%d")
        end_date = datetime.strptime(year + bi_monthly_range[1], "%Y%m%d")

        # Filter images within the bi-monthly range
        images_in_range = filter_images_by_date_range(df, start_date, end_date, year)
        if images_in_range:
            images_in_range = pd.concat(images_in_range)
            images_in_range = images_in_range.sort_values('nameCopy')

        print(f"-> Evaluating images from {start_date} to {end_date} ({len(images_in_range)} samples)")

        # Initialize counters for additional images from adjacent months
        index_next = 0
        index_prev = 0

        # If the number of images exceeds the threshold, filter the best ones
        if len(images_in_range) > configuration["images_per_group_threshold"]:
            
            # Define the start and end dates for the first month of the bi-monthly period
            monthly_range = configuration["monthly_ranges"][month_idx]
            start_date_monthly = datetime.strptime(year + monthly_range[0], "%Y%m%d")
            end_date_monthly = datetime.strptime(year + monthly_range[1], "%Y%m%d")

            # Get images for the first month only
            images_first_month = filter_images_by_date_range(images_in_range, start_date_monthly, end_date_monthly, year)
            images_first_month = pd.concat(images_first_month)

            missing_images = configuration["images_per_group_threshold"] - len(images_first_month)  # Calculate how many images are missing
            no_more_next = False  # Flag to stop checking subsequent months
            no_more_prev = False  # Flag to stop checking previous months

            print(f"  More data than {configuration['images_per_group_threshold']} -> There are {len(images_in_range)} images.")
            print(f"  Data available in first month (from {start_date_monthly} to {end_date_monthly}) are {len(images_first_month)}")

            # Try to fill missing images from adjacent months
            while missing_images > 0:
                # Fill from the next month if applicable
                if monthly_range[1] != "1231":

                    next_month_range = configuration["monthly_ranges"][month_idx + 1]
                    start_date_next = datetime.strptime(year + next_month_range[0], "%Y%m%d")
                    end_date_next = datetime.strptime(year + next_month_range[1], "%Y%m%d")
                    images_next_month = filter_images_by_date_range(images_in_range, start_date_next, end_date_next, year)

                    print(f"    Analyzing subsequent month ({next_month_range}) from {start_date_next} to {end_date_next} -> Has {len(images_next_month)} images.")

                    if images_next_month and not no_more_next:
                        images_next_month = pd.concat(images_next_month)
                        images_next_month = images_next_month.sort_values('nameCopy')
                        images_first_month = pd.concat([images_next_month.iloc[[index_next]], images_first_month])
                        index_next += 1
                        missing_images -= 1
                        if index_next == len(images_next_month):
                            no_more_next = True
                    else:
                        no_more_next = True

                # Fill from the previous month if applicable
                if monthly_range[0] != "0101":
                    
                    prev_month_range = configuration["monthly_ranges"][month_idx - 1]
                    start_date_prev = datetime.strptime(year + prev_month_range[0], "%Y%m%d")
                    end_date_prev = datetime.strptime(year + prev_month_range[1], "%Y%m%d")
                    images_prev_month = filter_images_by_date_range(images_in_range, start_date_prev, end_date_prev, year)

                    print(f"    Analyzing previous month ({prev_month_range}) from {start_date_prev} to {end_date_prev} -> Has {len(images_prev_month)} images.")

                    if images_prev_month and not no_more_prev:
                        images_prev_month = pd.concat(images_prev_month)
                        images_prev_month = images_prev_month.sort_values('nameCopy', ascending=False)
                        images_first_month = pd.concat([images_prev_month.iloc[[index_prev]], images_first_month])
                        index_prev += 1
                        missing_images -= 1
                        if index_prev == len(images_prev_month):
                            no_more_prev = True
                    else:
                        no_more_prev = True

                if no_more_next and no_more_prev:
                    break

            # If the number of images exceeds the threshold, keep only the top IMAGES_PER_GROUP_THRESHOLD
            if len(images_first_month) > configuration["images_per_group_threshold"]:
                images_first_month = images_first_month.sort_values('cloudCover')
                images_first_month = images_first_month[:configuration["images_per_group_threshold"]]

            # Ensure all orbits are represented in the selected images
            current_orbits = images_first_month['orbit'].unique()
            if len(current_orbits) != len(all_orbits):
                for orbit in all_orbits:
                    if orbit not in current_orbits:
                        orbit_images = images_in_range[images_in_range["orbit"] == orbit]
                        if not orbit_images.empty:
                            orbit_images = orbit_images.sort_values('nodata_percentage')
                            images_first_month = pd.concat([orbit_images.iloc[[0]], images_first_month])

            print(f"  Found {len(images_first_month)} images.")
            final_images.append(images_first_month)

        else:
            final_images.append(images_in_range)

    return final_images  # Return the final set of selected images


if __name__ == '__main__':

    configuration = retrieve_configuration()
    print_map(configuration)

    result_images = reduce_number_of_images(configuration)

    for df in result_images:
        if len(df) > 0:
            print_dataframe(df.sort_values(by=['timestamp']))
