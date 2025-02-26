import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re

def read_battery_metadata(file_path):
    """
    Read and parse the metadata.csv file containing battery test data.
    
    Args:
        file_path (str): Path to the metadata.csv file
        
    Returns:
        pd.DataFrame: Processed DataFrame with battery test data
    """
    # Read the file as text first to handle the irregular format
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract headers from the first line
    headers = lines[0].strip().split(',')
    
    # Initialize list to store parsed data
    data = []
    
    # Process each line (excluding header)
    for line in lines[1:]:
        # Split by comma
        values = line.strip().split(',')
        
        if len(values) >= len(headers):
            # Create a dictionary for this row
            row_dict = {}
            
            # Process each column
            for i, header in enumerate(headers):
                if i < len(values):
                    # Handle start_time which is stored as an array
                    if header == 'start_time' and '[' in values[i]:
                        # Extract the array values
                        time_str = ''.join(values[i:i+6])  # Join parts that might be split
                        # Use regex to extract numbers from the array notation
                        time_values = re.findall(r'[-+]?\d*\.\d+|\d+', time_str)
                        if len(time_values) >= 6:
                            # Convert to proper datetime format (assuming [year, month, day, hour, minute, second])
                            try:
                                year = int(float(time_values[0]))
                                month = int(float(time_values[1]))
                                day = int(float(time_values[2]))
                                hour = int(float(time_values[3]))
                                minute = int(float(time_values[4]))
                                second = float(time_values[5])
                                
                                # Create datetime string in ISO format
                                datetime_str = f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{int(second):02d}"
                                row_dict[header] = datetime_str
                            except (ValueError, IndexError):
                                row_dict[header] = None
                        else:
                            row_dict[header] = None
                    else:
                        # Handle numeric values
                        try:
                            value = values[i].strip()
                            # Try to convert to float if it looks like a number
                            if value and value != 'nan':
                                try:
                                    row_dict[header] = float(value)
                                except ValueError:
                                    row_dict[header] = value
                            else:
                                row_dict[header] = None
                        except:
                            row_dict[header] = None
            
            # Add row to data list
            data.append(row_dict)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def extract_eis_data(metadata_df):
    """
    Extract EIS (Electrochemical Impedance Spectroscopy) data from the metadata.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame containing battery metadata
        
    Returns:
        pd.DataFrame: DataFrame containing only EIS measurements
    """
    # Filter for impedance measurements
    eis_df = metadata_df[metadata_df['type'] == 'impedance'].copy()
    
    # Make sure Re and Rct columns are numeric
    for col in ['Re', 'Rct']:
        if col in eis_df.columns:
            eis_df[col] = pd.to_numeric(eis_df[col], errors='coerce')
    
    # Sort by battery_id and test_id to ensure chronological order
    eis_df = eis_df.sort_values(by=['battery_id', 'test_id'])
    
    # Add a cycle number approximation based on test_id
    eis_df['cycle'] = eis_df.groupby('battery_id')['test_id'].transform(
        lambda x: (x - x.min()) // 2
    )
    
    return eis_df

def plot_battery_eis_3d(eis_df, battery_id=None):
    """
    Create a 3D plot of impedance data for a specific battery or all batteries.
    
    Args:
        eis_df (pd.DataFrame): DataFrame containing EIS measurements
        battery_id (str, optional): Battery ID to plot. If None, plots data for all batteries.
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Filter for specific battery if requested
    if battery_id:
        plot_df = eis_df[eis_df['battery_id'] == battery_id].copy()
        title_suffix = f" for Battery {battery_id}"
    else:
        plot_df = eis_df.copy()
        title_suffix = " for All Batteries"
    
    # Check if we have data to plot
    if len(plot_df) == 0:
        print(f"No EIS data found{' for battery ' + battery_id if battery_id else ''}")
        return None
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get unique battery IDs for separate colors
    batteries = plot_df['battery_id'].unique()
    
    # Define colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(plot_df['cycle'].min(), plot_df['cycle'].max())
    
    # Plot data for each battery
    for battery in batteries:
        battery_data = plot_df[plot_df['battery_id'] == battery]
        cycles = sorted(battery_data['cycle'].unique())
        
        # Plot each cycle's data
        for cycle in cycles:
            cycle_data = battery_data[battery_data['cycle'] == cycle]
            if len(cycle_data) > 0:
                color = cmap(norm(cycle))
                
                # Plot as scatter points
                ax.scatter(
                    cycle_data['Re'], 
                    -cycle_data['Rct'],  # Negate for conventional EIS display
                    cycle,
                    color=color, 
                    s=50, 
                    label=f'{battery} - Cycle {cycle}' if cycle == cycles[0] else ""
                )
                
                # Connect points if we have multiple for this cycle
                if len(cycle_data) > 1:
                    ax.plot(
                        cycle_data['Re'], 
                        -cycle_data['Rct'], 
                        cycle, 
                        color=color, 
                        linestyle='-', 
                        linewidth=2
                    )
    
    # Set axis labels and title
    ax.set_xlabel('Re(Z) (kΩ)', fontsize=12)
    ax.set_ylabel('-Rct (kΩ)', fontsize=12)  # Using Rct as Im(Z)
    ax.set_zlabel('Cycle Number', fontsize=12)
    ax.set_title(f'3D Plot of Battery Impedance Change with Aging{title_suffix}', fontsize=14)
    
    # Add a color bar for cycle progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Cycle Number', fontsize=12)
    
    # Add legend if we're showing multiple batteries
    if len(batteries) > 1:
        ax.legend(title="Batteries")
    
    # Adjust view angle
    ax.view_init(elev=30, azim=45)
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def analyze_all_batteries(file_path):
    """
    Read metadata, process EIS data, and generate visualizations for all batteries.
    
    Args:
        file_path (str): Path to the metadata.csv file
    """
    # Read and process the metadata
    print(f"Reading metadata from {file_path}...")
    metadata_df = read_battery_metadata(file_path)
    
    # Basic info about the dataset
    battery_count = metadata_df['battery_id'].nunique()
    print(f"Found data for {battery_count} batteries")
    
    # Extract EIS data
    eis_df = extract_eis_data(metadata_df)
    print(f"Extracted {len(eis_df)} EIS measurements")
    
    # List all batteries
    batteries = sorted(metadata_df['battery_id'].unique())
    print(f"Batteries in dataset: {', '.join(batteries)}")
    
    # Create output directory for plots if it doesn't exist
    output_dir = "battery_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate overall plot for all batteries
    overall_fig = plot_battery_eis_3d(eis_df)
    if overall_fig:
        overall_fig.savefig(os.path.join(output_dir, "all_batteries_eis_3d.png"))
        plt.close(overall_fig)
    
    # Generate individual plots for each battery
    for battery in batteries:
        print(f"Generating plot for battery {battery}...")
        battery_fig = plot_battery_eis_3d(eis_df, battery)
        if battery_fig:
            battery_fig.savefig(os.path.join(output_dir, f"{battery}_eis_3d.png"))
            plt.close(battery_fig)
    
    print(f"All plots saved to {output_dir}/")

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    metadata_file = "metadata.csv"
    
    try:
        analyze_all_batteries(metadata_file)
    except Exception as e:
        print(f"Error processing metadata: {e}")