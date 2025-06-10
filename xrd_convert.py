import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
from datetime import datetime, timedelta
import re
import zipfile
import struct


def run_data_converter():
    def extract_key_ras_metadata(metadata_dict):
        key_metadata = {
            'X-ray Target': metadata_dict.get('HW_XG_TARGET_NAME', 'N/A'),
            'Voltage (kV)': metadata_dict.get('MEAS_COND_XG_VOLTAGE', 'N/A'),
            'Current (mA)': metadata_dict.get('MEAS_COND_XG_CURRENT', 'N/A'),
            'K-Alpha1 (Å)': metadata_dict.get('HW_XG_WAVE_LENGTH_ALPHA1', 'N/A'),
            'Scan Axis': metadata_dict.get('MEAS_SCAN_AXIS_X', 'N/A'),
            'Start Angle (°)': metadata_dict.get('MEAS_SCAN_START', 'N/A'),
            'Stop Angle (°)': metadata_dict.get('MEAS_SCAN_STOP', 'N/A'),
            'Step Size (°)': metadata_dict.get('MEAS_SCAN_STEP', 'N/A'),
            'Scan Speed': f"{metadata_dict.get('MEAS_SCAN_SPEED', 'N/A')} {metadata_dict.get('MEAS_SCAN_SPEED_UNIT', '')}".strip(),
            'Divergence Slit (DS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-18', 'N/A'),
            'Scattering Slit (SS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-35', 'N/A'),
            'Receiving Slit (RS)': metadata_dict.get('MEAS_COND_AXIS_POSITION-42', 'N/A')
        }
        return key_metadata

    def parse_xrdml(file_content):
        try:
            root = ET.fromstring(file_content)
            namespace = ''
            if '}' in root.tag:
                namespace = root.tag.split('}')[0][1:]
            ns = {'xrd': namespace}

            def find_text(path, default='N/A'):
                element = root.find(path, ns)
                return element.text if element is not None else default

            def find_attrib(path, attribute, default='N/A'):
                element = root.find(path, ns)
                return element.attrib.get(attribute, default) if element is not None else default

            metadata = {
                'Status': find_attrib('xrd:xrdMeasurement', 'status'),
                'Measurement Type': find_attrib('xrd:xrdMeasurement', 'measurementType'),
                'Start Time': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:startTimeStamp'),
                'End Time': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:endTimeStamp'),
                'Author': find_text('xrd:xrdMeasurement/xrd:scan/xrd:header/xrd:author/xrd:name'),
                'Anode Material': find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:anodeMaterial'),
                'X-ray Tube Tension': f"{find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:tension')} {find_attrib('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:tension', 'unit')}",
                'X-ray Tube Current': f"{find_text('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:current')} {find_attrib('xrd:xrdMeasurement/xrd:incidentBeamPath/xrd:xRayTube/xrd:current', 'unit')}",
                'K-Alpha1 Wavelength (Å)': find_text('xrd:xrdMeasurement/xrd:usedWavelength/xrd:kAlpha1'),
                'Detector': find_attrib('xrd:xrdMeasurement/xrd:diffractedBeamPath/xrd:detector', 'name'),
                'Scan Axis': find_attrib('xrd:xrdMeasurement/xrd:scan', 'scanAxis'),
            }
            data_points_path = 'xrd:xrdMeasurement/xrd:scan/xrd:dataPoints'
            start_pos_2theta = float(find_text(f'{data_points_path}/xrd:positions[@axis="2Theta"]/xrd:startPosition'))
            end_pos_2theta = float(find_text(f'{data_points_path}/xrd:positions[@axis="2Theta"]/xrd:endPosition'))
            intensities_str = find_text(f'{data_points_path}/xrd:intensities')
            intensities = np.array(intensities_str.split(), dtype=float)
            two_theta_array = np.linspace(start_pos_2theta, end_pos_2theta, len(intensities))
            data_df = pd.DataFrame({'2Theta': two_theta_array, 'Intensity': intensities})
            return metadata, data_df
        except Exception as e:
            st.error(f"Failed to parse XRDML file. Error: {e}")
            return None, None

    def parse_ras(file_content):
        try:
            full_metadata = {}
            data_lines = []
            in_header_section = False
            in_data_section = False

            for line in file_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line == '*RAS_HEADER_START':
                    in_header_section = True
                    continue
                if line == '*RAS_HEADER_END':
                    in_header_section = False
                    continue
                if line == '*RAS_INT_START':
                    in_data_section = True
                    continue
                if line == '*RAS_INT_END':
                    in_data_section = False
                    break

                if in_header_section and line.startswith('*'):
                    parts = line[1:].split(None, 1)
                    if len(parts) == 2:
                        key, value = parts
                        full_metadata[key] = value.strip('"')

                if in_data_section:
                    data_parts = line.split()
                    if len(data_parts) >= 2:
                        try:
                            angle = float(data_parts[0])
                            intensity = float(data_parts[1])
                            data_lines.append([angle, intensity])
                        except ValueError:
                            continue

            if not data_lines:
                st.error("No data points found in the RAS file.")
                return None, None, None

            data_df = pd.DataFrame(data_lines, columns=['2Theta', 'Intensity'])
            key_metadata = extract_key_ras_metadata(full_metadata)
            return full_metadata, key_metadata, data_df

        except Exception as e:
            st.error(f"Failed to parse RAS file. Error: {e}")
            return None, None, None

    def parse_rasx(uploaded_file_object):
        try:
            full_metadata = {}
            data_df = None
            with zipfile.ZipFile(uploaded_file_object, 'r') as zf:
                xml_filename = next((f for f in zf.namelist() if f.lower().endswith('.xml')), None)
                if not xml_filename:
                    st.error("No XML metadata file found in the RASX archive.")
                    return None, None, None

                xml_bytes = zf.read(xml_filename)
                xml_content_str = xml_bytes.decode('shift_jis', errors='replace')
                root = ET.fromstring(xml_content_str)

                for param in root.iter('Parameter'):
                    if 'name' in param.attrib and 'value' in param.attrib:
                        full_metadata[param.attrib['name']] = param.attrib['value']

                bin_filename = next((f for f in zf.namelist() if f.lower().endswith('.bin')), None)
                asc_filename = next((f for f in zf.namelist() if f.lower().endswith('.asc')), None)

                if bin_filename:
                    binary_content = zf.read(bin_filename)
                    intensities = np.frombuffer(binary_content, dtype=np.float32)
                    num_points = len(intensities)
                    start_angle = float(full_metadata.get('Start', 0))
                    stop_angle = float(full_metadata.get('Stop', 90))
                    angles = np.linspace(start_angle, stop_angle, num_points)
                    data_df = pd.DataFrame({'2Theta': angles, 'Intensity': intensities})
                elif asc_filename:
                    asc_content = zf.read(asc_filename).decode('utf-8', errors='replace')
                    data_df = parse_xy(asc_content)
                else:
                    st.error("No data file (.bin or .asc) found in the RASX archive.")
                    return None, None, None

            key_metadata = extract_key_ras_metadata(full_metadata)
            return full_metadata, key_metadata, data_df
        except Exception as e:
            st.error(f"Failed to parse RASX file. Error: {e}")
            return None, None, None

    def parse_raw_v1(file_content_bytes):
        metadata = {}
        header_size = 1024
        file_size = len(file_content_bytes)
        if 'debug_messages' not in st.session_state:
            st.session_state.debug_messages = []

        try:
            num_points = (file_size - header_size) // 4
            if num_points <= 0:
                st.error("Invalid file size for RAW 1.01 format.")
                return None, None
            metadata['Number of Points'] = num_points

            st.session_state.debug_messages.append(
                f"DEBUG: File size: {file_size} bytes, Header size: {header_size} bytes, Data points: {num_points}")
            step_size = None
            start_angle = None
            end_angle = None

            st.session_state.debug_messages.append("DEBUG: Scanning for step size information...")

            step_size_candidates = []
            for offset in range(200, min(1000, file_size - 4), 4):
                try:
                    value = struct.unpack_from('<f', file_content_bytes, offset)[0]
                    if 0.0001 <= value <= 0.5 and not np.isnan(value) and not np.isinf(value):
                        step_size_candidates.append((offset, value))
                except (struct.error, ValueError):
                    continue

            if step_size_candidates:
                expected_step = 0.4 / (num_points - 1) if num_points > 1 else 0.01

                best_step = min(step_size_candidates, key=lambda x: abs(x[1] - expected_step))
                if abs(best_step[1] - expected_step) < expected_step * 0.5:  # Within 50% of expected
                    step_size = best_step[1]

            st.session_state.debug_messages.append("DEBUG: Scanning for start and end angles...")

            potential_offsets = [
                924, 920, 916, 912, 908, 904, 900,
                568, 564, 560, 556, 552, 548, 544,
                300, 304, 308, 312, 316, 320,
                400, 404, 408, 412, 416, 420,
                500, 504, 508, 512, 516, 520,
            ]

            found_candidates = []

            for offset in potential_offsets:
                if offset + 4 <= file_size:
                    try:
                        value = struct.unpack_from('<f', file_content_bytes, offset)[0]
                        if -10.0 <= value <= 10.0 and not np.isnan(value) and not np.isinf(value):
                            found_candidates.append((offset, value))
                    except (struct.error, ValueError):
                        continue
            if step_size is not None:
                total_range = step_size * (num_points - 1)
                if found_candidates:
                    for offset, candidate_start in found_candidates:
                        candidate_end = candidate_start + total_range
                        if (-1.0 <= candidate_start <= 1.0 and -1.0 <= candidate_end <= 1.0):
                            start_angle = candidate_start
                            end_angle = candidate_end
                            break
                if start_angle is None:
                    for offset, candidate_end in found_candidates:
                        candidate_start = candidate_end - total_range
                        if (-1.0 <= candidate_start <= 1.0 and -1.0 <= candidate_end <= 1.0):
                            start_angle = candidate_start
                            end_angle = candidate_end
                            break
            if start_angle is None or end_angle is None:
                if len(found_candidates) >= 2:
                    found_candidates.sort(key=lambda x: x[1])
                    target_start = -0.2
                    target_end = 0.2
                    best_start = min(found_candidates, key=lambda x: abs(x[1] - target_start))
                    best_end = min(found_candidates, key=lambda x: abs(x[1] - target_end))

                    if abs(best_start[1] - target_start) < 0.1:
                        start_angle = best_start[1]
                        metadata['Start Angle (Omega)'] = f"{start_angle:.6f}"
                    if abs(best_end[1] - target_end) < 0.1:
                        end_angle = best_end[1]
                        metadata['End Angle (Omega)'] = f"{end_angle:.6f}"

                if start_angle is None or end_angle is None:
                    for offset in range(400, min(1000, file_size - 8), 4):
                        try:
                            val1 = struct.unpack_from('<f', file_content_bytes, offset)[0]
                            val2 = struct.unpack_from('<f', file_content_bytes, offset + 4)[0]
                            if (-1.0 <= val1 <= 1.0 and -1.0 <= val2 <= 1.0 and
                                    not np.isnan(val1) and not np.isnan(val2) and
                                    not np.isinf(val1) and not np.isinf(val2) and
                                    val1 != val2):

                                range_size = abs(val2 - val1)
                                if 0.1 <= range_size <= 2.0:
                                    start_angle = min(val1, val2)
                                    end_angle = max(val1, val2)
                                    metadata['Start Angle (Omega)'] = f"{start_angle:.6f}"
                                    metadata['End Angle (Omega)'] = f"{end_angle:.6f}"
                                    break
                        except (struct.error, ValueError):
                            continue
            if start_angle is None:
                start_angle = -0.2
                metadata['Start Angle (Omega)'] = f"{start_angle:.6f} (default)"

            if end_angle is None:
                end_angle = 0.2
                metadata['End Angle (Omega)'] = f"{end_angle:.6f} (default)"
            if step_size is None:
                if start_angle is not None and end_angle is not None:
                    step_size = (end_angle - start_angle) / (num_points - 1) if num_points > 1 else 0
                else:
                    step_size = 0.4 / (num_points - 1) if num_points > 1 else 0.01
            metadata['Step Size (Omega, Calculated)'] = f"{step_size:.6f}"
            try:
                fixed_2theta = struct.unpack_from('<f', file_content_bytes, offset=568)[0]
                if not np.isnan(fixed_2theta) and not np.isinf(fixed_2theta) and abs(fixed_2theta) < 180:
                    metadata['Fixed 2-Theta Angle'] = f"{fixed_2theta:.4f}"
                else:
                    metadata['Fixed 2-Theta Angle'] = 'N/A'
            except (struct.error, IndexError):
                metadata['Fixed 2-Theta Angle'] = 'N/A'
            try:
                target_name_bytes = struct.unpack_from('2s', file_content_bytes, offset=608)[0]
                target_name = target_name_bytes.decode('utf-8', errors='ignore').strip('\x00').strip()
                metadata['X-ray Target'] = target_name if target_name else 'N/A'
            except (struct.error, IndexError):
                metadata['X-ray Target'] = 'N/A'
            if step_size is not None and start_angle is not None:
                angles = np.arange(num_points) * step_size + start_angle
            elif start_angle is not None and end_angle is not None:
                angles = np.linspace(start_angle, end_angle, num_points)
            else:
                default_start = -0.2
                angles = np.arange(num_points) * step_size + default_start
                start_angle = default_start
                end_angle = angles[-1]

            metadata['Start Angle (Omega)'] = f"{start_angle:.6f}"
            metadata['End Angle (Omega)'] = f"{end_angle:.6f}"
            try:
                intensities = np.frombuffer(
                    file_content_bytes,
                    dtype=np.float32,
                    count=num_points,
                    offset=header_size
                )
                if len(intensities) != num_points:
                    st.error(f"DEBUG: Expected {num_points} intensities, got {len(intensities)}")
                    return None, None
            except Exception as e:
                st.error(f"DEBUG: Failed to read intensity data: {e}")
                return None, None
            data_df = pd.DataFrame({'2Theta': angles, 'Intensity': intensities})
            st.info("Omega scan data has been loaded. The scan axis (Omega) is displayed as '2Theta' in the plot.")

            with st.expander("Show Parser Debugging Info"):
                st.code("\n".join(st.session_state.debug_messages), language='text')
            st.session_state.debug_messages = []

            return metadata, data_df

        except Exception as e:
            st.error(f"Failed to parse RAW 1.01 file. Error: {e}")
            import traceback
            st.error(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
            return None, None

    def parse_raw_v4(file_content_bytes):
        metadata = {}
        file_size = len(file_content_bytes)
        data_offset = 2600

        try:
            try:
                metadata['Start Angle (°)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=136)[0]:.4f}"
            except (struct.error, IndexError):
                metadata['Start Angle (°)'] = 'N/A'
            try:
                metadata['Step Size (°)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=140)[0]:.4f}"
            except (struct.error, IndexError):
                metadata['Step Size (°)'] = 'N/A'
            try:
                metadata['Time per Step (s)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=152)[0]:.2f}"
            except (struct.error, IndexError):
                metadata['Time per Step (s)'] = 'N/A'
            try:
                metadata['K-Alpha1 (Å)'] = f"{struct.unpack_from('<f', file_content_bytes, offset=308)[0]:.5f}"
            except (struct.error, IndexError):
                metadata['K-Alpha1 (Å)'] = 'N/A'
            try:
                target_name_bytes = struct.unpack_from('12s', file_content_bytes, offset=244)[0];
                metadata[
                    'X-ray Target'] = target_name_bytes.decode('utf-8', errors='ignore').strip('\x00').strip()
            except (struct.error, IndexError):
                metadata['X-ray Target'] = 'N/A'

            num_points = 0
            try:
                num_points = struct.unpack_from('<i', file_content_bytes, offset=148)[0]
            except (struct.error, IndexError):
                pass

            if not num_points or num_points <= 0:
                st.warning("Could not read points from v4 header. Calculating from file size.")
                data_size = file_size - data_offset
                if data_size < 0:
                    st.error("File is smaller than the expected v4 header size.")
                    return None, None
                num_points = data_size // 4

            metadata['Number of Points'] = num_points
            if num_points <= 0: return metadata, None

            if metadata['Start Angle (°)'] == 'N/A' or metadata['Step Size (°)'] == 'N/A': return metadata, None

            intensities = np.frombuffer(file_content_bytes, dtype=np.float32, count=num_points, offset=data_offset)
            angles = np.arange(num_points) * float(metadata['Step Size (°)']) + float(metadata['Start Angle (°)'])
            data_df = pd.DataFrame({'2Theta': angles, 'Intensity': intensities})

            return metadata, data_df
        except Exception as e:
            st.error(f"A critical error occurred while parsing the RAW v4 file. Error: {e}")
            return None, None

    def parse_raw(file_content_bytes):

        if file_content_bytes.startswith(b'RAW1.01'):
            st.success("Assuming Bruker RAW 1.01 file format.")
            return parse_raw_v1(file_content_bytes)
        else:
            st.success("Assuming Bruker RAW v4 file format.")
            return parse_raw_v4(file_content_bytes)

    def parse_xy(file_content):
        try:
            lines = file_content.splitlines()
            if not lines:
                st.error("The XY file is empty.")
                return None
            first_line = lines[0]
            has_header = any(char.isalpha() for char in first_line)
            data_io = StringIO(file_content)
            skiprows = 1 if has_header else 0
            df = pd.read_csv(data_io, sep=r'[\s,;]+', engine='python', header=None, skiprows=skiprows,
                             names=['2Theta', 'Intensity'], comment='#')
            return df.dropna().astype(float)
        except Exception as e:
            st.error(f"Failed to parse XY file. Error: {e}")
            return None

    def convert_to_xy(data_df, include_header=False):
        try:
            output = StringIO()
            output_df = data_df[['2Theta', 'Intensity']]

            header = ['2Theta', 'Intensity'] if include_header else False
            output_df.to_csv(output, sep='\t', header=header, index=False, float_format='%.6f')
            return output.getvalue()
        except KeyError:
            st.error("The DataFrame is missing the required '2Theta' or 'Intensity' columns.")
            return ""

    def generate_xrdml(metadata_df, data_df, filename=""):
        meta_dict = pd.Series(metadata_df.Value.values, index=metadata_df.Parameter).to_dict()
        start_2theta = data_df['2Theta'].min()
        end_2theta = data_df['2Theta'].max()
        intensities_str = ' '.join(map(lambda x: f"{x:.3f}", data_df['Intensity'].values))

        sample_name = filename.split('.')[0] if filename else meta_dict.get('Sample Name', 'Converted Sample')

        return f"""<?xml version="1.0" encoding="utf-8" standalone="no"?>
<xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.3" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.xrdml.com/XRDMeasurement/1.3 http://www.xrdml.com/XRDMeasurement/1.3/XRDMeasurement.xsd" status="{meta_dict.get('Status', 'Completed')}">
  <sample type="To be analyzed">
    <id>{meta_dict.get('Sample ID', 'N/A')}</id>
    <name>{sample_name}</name>
  </sample>
  <xrdMeasurement measurementType="{meta_dict.get('Measurement Type', 'Scan')}" status="{meta_dict.get('Status', 'Completed')}">
    <usedWavelength intended="K-Alpha">
      <kAlpha1 unit="Angstrom">{meta_dict.get('K-Alpha1 Wavelength (Å)', '1.54056')}</kAlpha1>
      <kAlpha2 unit="Angstrom">{meta_dict.get('K-Alpha2 Wavelength (Å)', '1.54439')}</kAlpha2>
      <kBeta unit="Angstrom">{meta_dict.get('K-Beta Wavelength (Å)', '1.39225')}</kBeta>
      <ratioKAlpha2KAlpha1>{meta_dict.get('Ratio K-Alpha2/K-Alpha1', '0.5')}</ratioKAlpha2KAlpha1>
    </usedWavelength>
    <incidentBeamPath>
      <xRayTube>
        <tension unit="kV">{meta_dict.get('X-ray Tube Tension (kV)', '45')}</tension>
        <current unit="mA">{meta_dict.get('X-ray Tube Current (mA)', '40')}</current>
        <anodeMaterial>{meta_dict.get('Anode Material', 'Cu')}</anodeMaterial>
      </xRayTube>
    </incidentBeamPath>
    <scan appendNumber="0" mode="{meta_dict.get('Scan Mode', 'Continuous')}" scanAxis="{meta_dict.get('Scan Axis', 'Gonio')}" status="Completed">
      <header>
        <startTimeStamp>{meta_dict.get('Start Time', datetime.now().isoformat())}</startTimeStamp>
        <endTimeStamp>{meta_dict.get('End Time', (datetime.now() + timedelta(minutes=30)).isoformat())}</endTimeStamp>
        <author>
          <name>{meta_dict.get('Author', 'XRDlicious User')}</name>
        </author>
        <source>
          <applicationSoftware version="1.0">{meta_dict.get('Application Software', 'XRDlicious')}</applicationSoftware>
        </source>
      </header>
      <dataPoints>
        <positions axis="2Theta" unit="deg">
          <startPosition>{start_2theta:.6f}</startPosition>
          <endPosition>{end_2theta:.6f}</endPosition>
        </positions>
        <commonCountingTime unit="seconds">{meta_dict.get('Common Counting Time (s)', '1.0')}</commonCountingTime>
        <intensities unit="counts">{intensities_str}</intensities>
      </dataPoints>
    </scan>
  </xrdMeasurement>
</xrdMeasurements>
"""

    def generate_ras(metadata_df, data_df):
        meta_dict = pd.Series(metadata_df.Value.values, index=metadata_df.Parameter).to_dict()
        start_angle = f"{data_df['2Theta'].min():.4f}"
        stop_angle = f"{data_df['2Theta'].max():.4f}"
        data_count = str(len(data_df))
        step_size = f"{abs(data_df['2Theta'].iloc[1] - data_df['2Theta'].iloc[0]):.4f}" if len(
            data_df) > 1 else "0.0100"

        try:
            start_iso = meta_dict.get('MEAS_SCAN_START_TIME', datetime.now().isoformat())
            end_iso = meta_dict.get('MEAS_SCAN_END_TIME', datetime.now().isoformat())
            start_dt = datetime.fromisoformat(start_iso)
            end_dt = datetime.fromisoformat(end_iso)
            formatted_start_time = start_dt.strftime('%m/%d/%Y %H:%M:%S')
            formatted_end_time = end_dt.strftime('%m/%d/%Y %H:%M:%S')
        except ValueError:
            now = datetime.now()
            formatted_start_time = now.strftime('%m/%d/%Y %H:%M:%S')
            formatted_end_time = (now + timedelta(minutes=20)).strftime('%m/%d/%Y %H:%M:%S')

        header_template = f"""*RAS_DATA_START
*RAS_HEADER_START
*DISP_LINE_COLOR "4294901760"
*FILE_COMMENT ""
*FILE_MD5 ""
*FILE_MEMO ""
*FILE_OPERATOR "{meta_dict.get('FILE_OPERATOR', 'Administrator')}"
*FILE_PACKAGE_NAME "Package_BB"
*FILE_PART_ID "GeneralMeasurement(BB)"
*FILE_SAMPLE ""
*FILE_SYSTEM_NAME "SmartLabXE"
*FILE_TYPE "RAS_RAW"
*FILE_USERGROUP "Administrators"
*FILE_VERSION "1"
*HW_ATTACHMENT_ID "ATT0025"
*HW_ATTACHMENT_NAME "Standard"
*HW_ATTACHMENT_NAME_INTERNAL "Standard"
*HW_COUNTER_ID-0 "CUT0051"
*HW_COUNTER_ID-2 "CMC0020"
*HW_COUNTER_MONOCHRO_ID "CMC0020"
*HW_COUNTER_MONOCHRO_NAME "None"
*HW_COUNTER_MONOCHRO_NAME_INTERNAL "None"
*HW_COUNTER_NAME_INTERNAL "HyPix3000(H)"
*HW_COUNTER_NAME-0 "HyPix3000(H)"
*HW_COUNTER_NAME-1 "None"
*HW_COUNTER_NAME-2 "None"
*HW_COUNTER_PIXEL_SIZE "0.1"
*HW_COUNTER_SELECT_NAME "HyPix3000(H)"
*HW_EXTERNAL_CONTROLLER_NAME "None"
*HW_EXTERNAL_CONTROLLER_NAME_INTERNAL "None"
*HW_GONIOMETER_ID "GON0022"
*HW_GONIOMETER_NAME "StandardInplane"
*HW_GONIOMETER_NAME_INTERNAL "StandardInplaneEnc"
*HW_GONIOMETER_RADIUS-0 "90.0"
*HW_GONIOMETER_RADIUS-1 "114.0"
*HW_GONIOMETER_RADIUS-2 "173.5"
*HW_GONIOMETER_RADIUS-3 "300.0"
*HW_GONIOMETER_RADIUS-4 "187.0"
*HW_GONIOMETER_RADIUS-5 "300.0"
*HW_GONIOMETER_RADIUS-6 "113.0"
*HW_GONIOMETER_RADIUS-7 "331.0"
*HW_I_CBO_ID "CBO0021"
*HW_I_CBO_NAME "CBO"
*HW_I_CBO_NAME_INTERNAL "CBO"
*HW_I_MONOCHRO_ID "ISO0021"
*HW_I_MONOCHRO_NAME "IPS_adaptor"
*HW_I_MONOCHRO_NAME_INTERNAL "IPS_adaptor"
*HW_I_OPT_ID-1 "CBO0021"
*HW_I_PRIMARY_NAME_INTERNAL "Standard"
*HW_I_SLIT_NAME_INTERNAL "AutoIntegrated"
*HW_R_ATTENUATER_AUTOMODE "0"
*HW_R_ATTENUATER_ID "ATN0020"
*HW_R_ATTENUATER_NAME "No_unit"
*HW_R_ATTENUATOR_NAME_INTERNAL "No_unit"
*HW_R_OPT_ID-0 "RSS0022"
*HW_R_OPT_ID-1 "RCR0022"
*HW_R_OPT_ID-2 "RSO0021"
*HW_R_OPT_ID-3 "RRS0022"
*HW_R_OPT_ID-4 "ATN0020"
*HW_R_ROD_ID "RCR0022"
*HW_R_ROD_NAME "ROD_adaptor"
*HW_R_ROD_NAME_INTERNAL "ROD_adaptor"
*HW_R_RPS_ID "RSO0021"
*HW_R_RPS_NAME "RPS_adaptor"
*HW_R_RPS_NAME_INTERNAL "RPS_adaptor"
*HW_R_RS_ID "RRS0022"
*HW_R_RS_NAME "Virtual"
*HW_R_RS_NAME_INTERNAL "Virtual"
*HW_R_SS_ID "RSS0022"
*HW_R_SS_NAME "Auto_Zr"
*HW_R_SS_NAME_INTERNAL "Auto_Zr"
*HW_ROBOT_NAME "No_unit"
*HW_ROBOT_NAME_INTERNAL "No_unit"
*HW_SAMPLE_CAMERA_NAME "SampleCamera_Inp"
*HW_SAMPLE_CAMERA_NAME_INTERNAL "SampleCamera_Inp"
*HW_SAMPLE_HOLDER_ID "SMP0021"
*HW_SAMPLE_HOLDER_NAME "Z_ChiPhi"
*HW_SAMPLE_HOLDER_NAME_INTERNAL "Z_ChiPhi"
*HW_SAMPLE_NAME "PowderSample"
*HW_SAMPLE_NAME_INTERNAL "PowderSample"
*HW_SAMPLE_PLATE_NAME "ForWafer"
*HW_SAMPLE_PLATE_NAME_INTERNAL "ForWafer"
*HW_SAMPLE_SPACER_NAME "3-6mm"
*HW_SAMPLE_SPACER_NAME_INTERNAL "3-6mm"
*HW_USERINFO_CATALOG_NO ""
*HW_USERINFO_INSTRUMENT_ID ""
*HW_USERINFO_INSTRUMENT_NAME ""
*HW_USERINFO_INSTRUMENT_NO ""
*HW_USERINFO_MODEL ""
*HW_USERINFO_ORDER_NO ""
*HW_USERINFO_SERIAL_NO ""
*HW_USERINFO_VERIFY_INSTRUMENT_ID "00-07-fe-01-26-9c"
*HW_VER_RCD_COUNTER_UNIT "N/A"
*HW_VER_RCD_CPU "3.3.7"
*HW_VER_RCD_FPGA "1.2.3"
*HW_VER_RCD_GONIO_UNIT "6.4.0"
*HW_VER_RCD_INCIDENT_UNIT "6.4.0"
*HW_VER_RCD_RECEIVING_UNIT "8.0.1"
*HW_VER_RCD_TYPE "RINC"
*HW_VER_XGC_CPU "3.3.7"
*HW_VER_XGC_CW "3.0.8"
*HW_VER_XGC_HV "3.0.8"
*HW_VER_XGC_RS1 "3.0.8"
*HW_VER_XGC_RS2 "N/A"
*HW_VER_XGC_RT "N/A"
*HW_VER_XGC_TYPE "RINC"
*HW_XG_CURRENT_UNIT "mA"
*HW_XG_FOCUS "0.4mm x 8mm"
*HW_XG_FOCUS_TYPE "Fine"
*HW_XG_TARGET_ATOMIC_NUMBER "29"
*HW_XG_TARGET_NAME "{meta_dict.get('HW_XG_TARGET_NAME', 'Cu')}"
*HW_XG_TYPE "Hermetic"
*HW_XG_VOLTAGE_UNIT "kV"
*HW_XG_WAVE_LENGTH_ALPHA1 "1.540593"
*HW_XG_WAVE_LENGTH_ALPHA2 "1.544414"
*HW_XG_WAVE_LENGTH_BETA "1.392246"
*HW_XG_WAVE_LENGTH_UNIT "Angstrom"
*MEAS_COND_AXIS_NAME_INTERNAL-0 "ThetaS"
*MEAS_COND_AXIS_NAME_INTERNAL-1 "ThetaD"
*MEAS_COND_AXIS_NAME_INTERNAL-10 "PrimaryGeometry"
*MEAS_COND_AXIS_NAME_INTERNAL-100 "IncidentMonochromator"
*MEAS_COND_AXIS_NAME_INTERNAL-101 "ReceivingOptics"
*MEAS_COND_AXIS_NAME_INTERNAL-102 "CounterMonochromatorKind"
*MEAS_COND_AXIS_NAME_INTERNAL-11 "PrimaryMirrorType"
*MEAS_COND_AXIS_NAME_INTERNAL-12 "CBOType"
*MEAS_COND_AXIS_NAME_INTERNAL-13 "CBO-M"
*MEAS_COND_AXIS_NAME_INTERNAL-14 "CBO"
*MEAS_COND_AXIS_NAME_INTERNAL-15 "IncidentSollerSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-16 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME_INTERNAL-17 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME_INTERNAL-18 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME_INTERNAL-19 "Zs"
*MEAS_COND_AXIS_NAME_INTERNAL-2 "TwoTheta"
*MEAS_COND_AXIS_NAME_INTERNAL-20 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-21 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-22 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-23 "InFilter"
*MEAS_COND_AXIS_NAME_INTERNAL-24 "Chi"
*MEAS_COND_AXIS_NAME_INTERNAL-25 "Phi"
*MEAS_COND_AXIS_NAME_INTERNAL-26 "Z"
*MEAS_COND_AXIS_NAME_INTERNAL-27 "Alpha"
*MEAS_COND_AXIS_NAME_INTERNAL-28 "Beta"
*MEAS_COND_AXIS_NAME_INTERNAL-29 "TwoThetaChiPhi"
*MEAS_COND_AXIS_NAME_INTERNAL-3 "Omega"
*MEAS_COND_AXIS_NAME_INTERNAL-30 "AlphaI"
*MEAS_COND_AXIS_NAME_INTERNAL-31 "BetaI"
*MEAS_COND_AXIS_NAME_INTERNAL-32 "TwoThetaB"
*MEAS_COND_AXIS_NAME_INTERNAL-33 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME_INTERNAL-34 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME_INTERNAL-35 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME_INTERNAL-36 "Zr"
*MEAS_COND_AXIS_NAME_INTERNAL-37 "Filter"
*MEAS_COND_AXIS_NAME_INTERNAL-38 "PSA"
*MEAS_COND_AXIS_NAME_INTERNAL-39 "ReceivingSollerSlit"
*MEAS_COND_AXIS_NAME_INTERNAL-4 "TwoThetaTheta"
*MEAS_COND_AXIS_NAME_INTERNAL-40 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME_INTERNAL-41 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME_INTERNAL-42 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME_INTERNAL-43 "ModuleSensorType"
*MEAS_COND_AXIS_NAME_INTERNAL-44 "GainMode"
*MEAS_COND_AXIS_NAME_INTERNAL-45 "PHA"
*MEAS_COND_AXIS_NAME_INTERNAL-46 "DetectorBeamStop"
*MEAS_COND_AXIS_NAME_INTERNAL-47 "DetectorFilter"
*MEAS_COND_AXIS_NAME_INTERNAL-48 "DedicatedHolder"
*MEAS_COND_AXIS_NAME_INTERNAL-49 "Target_TargetTime"
*MEAS_COND_AXIS_NAME_INTERNAL-5 "TwoThetaOmega"
*MEAS_COND_AXIS_NAME_INTERNAL-50 "Target_XrayOnTime"
*MEAS_COND_AXIS_NAME_INTERNAL-51 "Target_TMPTime"
*MEAS_COND_AXIS_NAME_INTERNAL-52 "Target_RPTime"
*MEAS_COND_AXIS_NAME_INTERNAL-53 "Target_FilamentTime"
*MEAS_COND_AXIS_NAME_INTERNAL-54 "Target_IG"
*MEAS_COND_AXIS_NAME_INTERNAL-55 "Target_GP"
*MEAS_COND_AXIS_NAME_INTERNAL-56 "Target_FC"
*MEAS_COND_AXIS_NAME_INTERNAL-57 "HVPS_Bias"
*MEAS_COND_AXIS_NAME_INTERNAL-58 "HVPS_HVPSTime"
*MEAS_COND_AXIS_NAME_INTERNAL-59 "HVPS_Type1Time"
*MEAS_COND_AXIS_NAME_INTERNAL-6 "OmegaTwoTheta"
*MEAS_COND_AXIS_NAME_INTERNAL-60 "HVPS_Type2Time"
*MEAS_COND_AXIS_NAME_INTERNAL-61 "HVPS_Type3Time"
*MEAS_COND_AXIS_NAME_INTERNAL-62 "CW_IERTime"
*MEAS_COND_AXIS_NAME_INTERNAL-63 "CW_ECTime"
*MEAS_COND_AXIS_NAME_INTERNAL-64 "CW_Flow1"
*MEAS_COND_AXIS_NAME_INTERNAL-65 "CW_Temperature1"
*MEAS_COND_AXIS_NAME_INTERNAL-66 "CW_Pressure1"
*MEAS_COND_AXIS_NAME_INTERNAL-67 "CW_PressureIn"
*MEAS_COND_AXIS_NAME_INTERNAL-68 "CW_PressureOut"
*MEAS_COND_AXIS_NAME_INTERNAL-69 "CW_Flow2"
*MEAS_COND_AXIS_NAME_INTERNAL-7 "TwoThetaChi"
*MEAS_COND_AXIS_NAME_INTERNAL-70 "CW_Temperature2"
*MEAS_COND_AXIS_NAME_INTERNAL-71 "CW_Pressure2"
*MEAS_COND_AXIS_NAME_INTERNAL-72 "RE_EnclosureTemp"
*MEAS_COND_AXIS_NAME_INTERNAL-73 "RE_EnclosureHummidity"
*MEAS_COND_AXIS_NAME_INTERNAL-74 "RE_CabinetTemp"
*MEAS_COND_AXIS_NAME_INTERNAL-75 "RE_RPTemp"
*MEAS_COND_AXIS_NAME_INTERNAL-76 "RE_ShutterAXrayOnTime"
*MEAS_COND_AXIS_NAME_INTERNAL-77 "RE_ShutterATimes"
*MEAS_COND_AXIS_NAME_INTERNAL-78 "RE_ShutterAOpenCloseTime"
*MEAS_COND_AXIS_NAME_INTERNAL-79 "RE_ShutterACloseOpenTime"
*MEAS_COND_AXIS_NAME_INTERNAL-8 "GonioDirectBeamStop"
*MEAS_COND_AXIS_NAME_INTERNAL-80 "RE_ShutterBXrayOnTime"
*MEAS_COND_AXIS_NAME_INTERNAL-81 "RE_ShutterBTimes"
*MEAS_COND_AXIS_NAME_INTERNAL-82 "RE_ShutterBOpenCloseTime"
*MEAS_COND_AXIS_NAME_INTERNAL-83 "RE_ShutterBCloseOpenTime"
*MEAS_COND_AXIS_NAME_INTERNAL-84 "RE_XrayWarningRamp"
*MEAS_COND_AXIS_NAME_INTERNAL-85 "RE_ShutterAInstallation"
*MEAS_COND_AXIS_NAME_INTERNAL-86 "RE_ShutterARamp"
*MEAS_COND_AXIS_NAME_INTERNAL-87 "RE_ShutterBInstallation"
*MEAS_COND_AXIS_NAME_INTERNAL-88 "RE_ShutterBRamp"
*MEAS_COND_AXIS_NAME_INTERNAL-89 "RE_ExtShutterCLS"
*MEAS_COND_AXIS_NAME_INTERNAL-9 "Ts"
*MEAS_COND_AXIS_NAME_INTERNAL-90 "RE_ExtXrayOFF"
*MEAS_COND_AXIS_NAME_INTERNAL-91 "RE_ExtSafetyCircuit"
*MEAS_COND_AXIS_NAME_INTERNAL-92 "Version_CPU"
*MEAS_COND_AXIS_NAME_INTERNAL-93 "Version_RS1"
*MEAS_COND_AXIS_NAME_INTERNAL-94 "Version_RS2"
*MEAS_COND_AXIS_NAME_INTERNAL-95 "Version_HV"
*MEAS_COND_AXIS_NAME_INTERNAL-96 "Version_CW"
*MEAS_COND_AXIS_NAME_INTERNAL-97 "Version_RT"
*MEAS_COND_AXIS_NAME_INTERNAL-98 "Lens"
*MEAS_COND_AXIS_NAME_INTERNAL-99 "IncidentPrimary"
*MEAS_COND_AXIS_NAME-0 "ThetaS"
*MEAS_COND_AXIS_NAME-1 "ThetaD"
*MEAS_COND_AXIS_NAME-10 "PrimaryGeometry"
*MEAS_COND_AXIS_NAME-100 "IncidentMonochromator"
*MEAS_COND_AXIS_NAME-101 "ReceivingOptics"
*MEAS_COND_AXIS_NAME-102 "CounterMonochromatorKind"
*MEAS_COND_AXIS_NAME-11 "PrimaryMirrorType"
*MEAS_COND_AXIS_NAME-12 "CBOType"
*MEAS_COND_AXIS_NAME-13 "CBO-M"
*MEAS_COND_AXIS_NAME-14 "CBO"
*MEAS_COND_AXIS_NAME-15 "IncidentSollerSlit"
*MEAS_COND_AXIS_NAME-16 "IncidentSlitBox"
*MEAS_COND_AXIS_NAME-17 "IncidentSlitBox-_Axis"
*MEAS_COND_AXIS_NAME-18 "SlitDS"
*MEAS_COND_AXIS_NAME-19 "Zs"
*MEAS_COND_AXIS_NAME-2 "TwoTheta"
*MEAS_COND_AXIS_NAME-20 "IncidentAxdSlit"
*MEAS_COND_AXIS_NAME-21 "LLS"
*MEAS_COND_AXIS_NAME-22 "DHLSlit"
*MEAS_COND_AXIS_NAME-23 "InFilter"
*MEAS_COND_AXIS_NAME-24 "Chi"
*MEAS_COND_AXIS_NAME-25 "Phi"
*MEAS_COND_AXIS_NAME-26 "Z"
*MEAS_COND_AXIS_NAME-27 "Alpha"
*MEAS_COND_AXIS_NAME-28 "Beta"
*MEAS_COND_AXIS_NAME-29 "TwoThetaChiPhi"
*MEAS_COND_AXIS_NAME-3 "Omega"
*MEAS_COND_AXIS_NAME-30 "AlphaI"
*MEAS_COND_AXIS_NAME-31 "BetaI"
*MEAS_COND_AXIS_NAME-32 "TwoThetaB"
*MEAS_COND_AXIS_NAME-33 "ReceivingSlitBox1"
*MEAS_COND_AXIS_NAME-34 "ReceivingSlitBox1-_Axis"
*MEAS_COND_AXIS_NAME-35 "SlitSS"
*MEAS_COND_AXIS_NAME-36 "Zr"
*MEAS_COND_AXIS_NAME-37 "Filter"
*MEAS_COND_AXIS_NAME-38 "PSA"
*MEAS_COND_AXIS_NAME-39 "ReceivingSollerSlit"
*MEAS_COND_AXIS_NAME-4 "TwoThetaTheta"
*MEAS_COND_AXIS_NAME-40 "ReceivingSlitBox2"
*MEAS_COND_AXIS_NAME-41 "ReceivingSlitBox2-_Axis"
*MEAS_COND_AXIS_NAME-42 "SlitRS"
*MEAS_COND_AXIS_NAME-43 "ModuleSensorType"
*MEAS_COND_AXIS_NAME-44 "GainMode"
*MEAS_COND_AXIS_NAME-45 "PHA"
*MEAS_COND_AXIS_NAME-46 "DetectorBeamStop"
*MEAS_COND_AXIS_NAME-47 "DetectorFilter"
*MEAS_COND_AXIS_NAME-48 "DedicatedHolder"
*MEAS_COND_AXIS_NAME-49 "Target_TargetTime"
*MEAS_COND_AXIS_NAME-5 "TwoThetaOmega"
*MEAS_COND_AXIS_NAME-50 "Target_XrayOnTime"
*MEAS_COND_AXIS_NAME-51 "Target_TMPTime"
*MEAS_COND_AXIS_NAME-52 "Target_RPTime"
*MEAS_COND_AXIS_NAME-53 "Target_FilamentTime"
*MEAS_COND_AXIS_NAME-54 "Target_IG"
*MEAS_COND_AXIS_NAME-55 "Target_GP"
*MEAS_COND_AXIS_NAME-56 "Target_FC"
*MEAS_COND_AXIS_NAME-57 "HVPS_Bias"
*MEAS_COND_AXIS_NAME-58 "HVPS_HVPSTime"
*MEAS_COND_AXIS_NAME-59 "HVPS_Type1Time"
*MEAS_COND_AXIS_NAME-6 "OmegaTwoTheta"
*MEAS_COND_AXIS_NAME-60 "HVPS_Type2Time"
*MEAS_COND_AXIS_NAME-61 "HVPS_Type3Time"
*MEAS_COND_AXIS_NAME-62 "CW_IERTime"
*MEAS_COND_AXIS_NAME-63 "CW_ECTime"
*MEAS_COND_AXIS_NAME-64 "CW_Flow1"
*MEAS_COND_AXIS_NAME-65 "CW_Temperature1"
*MEAS_COND_AXIS_NAME-66 "CW_Pressure1"
*MEAS_COND_AXIS_NAME-67 "CW_PressureIn"
*MEAS_COND_AXIS_NAME-68 "CW_PressureOut"
*MEAS_COND_AXIS_NAME-69 "CW_Flow2"
*MEAS_COND_AXIS_NAME-7 "TwoThetaChi"
*MEAS_COND_AXIS_NAME-70 "CW_Temperature2"
*MEAS_COND_AXIS_NAME-71 "CW_Pressure2"
*MEAS_COND_AXIS_NAME-72 "RE_EnclosureTemp"
*MEAS_COND_AXIS_NAME-73 "RE_EnclosureHummidity"
*MEAS_COND_AXIS_NAME-74 "RE_CabinetTemp"
*MEAS_COND_AXIS_NAME-75 "RE_RPTemp"
*MEAS_COND_AXIS_NAME-76 "RE_ShutterAXrayOnTime"
*MEAS_COND_AXIS_NAME-77 "RE_ShutterATimes"
*MEAS_COND_AXIS_NAME-78 "RE_ShutterAOpenCloseTime"
*MEAS_COND_AXIS_NAME-79 "RE_ShutterACloseOpenTime"
*MEAS_COND_AXIS_NAME-8 "GonioDirectBeamStop"
*MEAS_COND_AXIS_NAME-80 "RE_ShutterBXrayOnTime"
*MEAS_COND_AXIS_NAME-81 "RE_ShutterBTimes"
*MEAS_COND_AXIS_NAME-82 "RE_ShutterBOpenCloseTime"
*MEAS_COND_AXIS_NAME-83 "RE_ShutterBCloseOpenTime"
*MEAS_COND_AXIS_NAME-84 "RE_XrayWarningRamp"
*MEAS_COND_AXIS_NAME-85 "RE_ShutterAInstallation"
*MEAS_COND_AXIS_NAME-86 "RE_ShutterARamp"
*MEAS_COND_AXIS_NAME-87 "RE_ShutterBInstallation"
*MEAS_COND_AXIS_NAME-88 "RE_ShutterBRamp"
*MEAS_COND_AXIS_NAME-89 "RE_ExtShutterCLS"
*MEAS_COND_AXIS_NAME-9 "Ts"
*MEAS_COND_AXIS_NAME-90 "RE_ExtXrayOFF"
*MEAS_COND_AXIS_NAME-91 "RE_ExtSafetyCircuit"
*MEAS_COND_AXIS_NAME-92 "Version_CPU"
*MEAS_COND_AXIS_NAME-93 "Version_RS1"
*MEAS_COND_AXIS_NAME-94 "Version_RS2"
*MEAS_COND_AXIS_NAME-95 "Version_HV"
*MEAS_COND_AXIS_NAME-96 "Version_CW"
*MEAS_COND_AXIS_NAME-97 "Version_RT"
*MEAS_COND_AXIS_NAME-98 "Lens"
*MEAS_COND_AXIS_NAME-99 "IncidentPrimary"
*MEAS_COND_AXIS_OFFSET-0 "1.1167"
*MEAS_COND_AXIS_OFFSET-1 "-0.0708"
*MEAS_COND_AXIS_OFFSET-10 "0"
*MEAS_COND_AXIS_OFFSET-100 "0"
*MEAS_COND_AXIS_OFFSET-101 "0"
*MEAS_COND_AXIS_OFFSET-102 "0"
*MEAS_COND_AXIS_OFFSET-11 "0"
*MEAS_COND_AXIS_OFFSET-12 "0"
*MEAS_COND_AXIS_OFFSET-13 "0"
*MEAS_COND_AXIS_OFFSET-14 "0"
*MEAS_COND_AXIS_OFFSET-15 "0"
*MEAS_COND_AXIS_OFFSET-16 "0"
*MEAS_COND_AXIS_OFFSET-17 "0"
*MEAS_COND_AXIS_OFFSET-18 "0"
*MEAS_COND_AXIS_OFFSET-19 "0"
*MEAS_COND_AXIS_OFFSET-2 "1.0459"
*MEAS_COND_AXIS_OFFSET-20 "0"
*MEAS_COND_AXIS_OFFSET-21 "0"
*MEAS_COND_AXIS_OFFSET-22 "0"
*MEAS_COND_AXIS_OFFSET-23 "0"
*MEAS_COND_AXIS_OFFSET-24 "0"
*MEAS_COND_AXIS_OFFSET-25 "0"
*MEAS_COND_AXIS_OFFSET-26 "0"
*MEAS_COND_AXIS_OFFSET-27 "0"
*MEAS_COND_AXIS_OFFSET-28 "0"
*MEAS_COND_AXIS_OFFSET-29 "0"
*MEAS_COND_AXIS_OFFSET-3 "1.1167"
*MEAS_COND_AXIS_OFFSET-30 "0"
*MEAS_COND_AXIS_OFFSET-31 "0"
*MEAS_COND_AXIS_OFFSET-32 "0"
*MEAS_COND_AXIS_OFFSET-33 "0"
*MEAS_COND_AXIS_OFFSET-34 "0"
*MEAS_COND_AXIS_OFFSET-35 "0"
*MEAS_COND_AXIS_OFFSET-36 "0"
*MEAS_COND_AXIS_OFFSET-37 "0"
*MEAS_COND_AXIS_OFFSET-38 "0"
*MEAS_COND_AXIS_OFFSET-39 "0"
*MEAS_COND_AXIS_OFFSET-4 "0"
*MEAS_COND_AXIS_OFFSET-40 "0"
*MEAS_COND_AXIS_OFFSET-41 "0"
*MEAS_COND_AXIS_OFFSET-42 "0"
*MEAS_COND_AXIS_OFFSET-43 "0"
*MEAS_COND_AXIS_OFFSET-44 "0"
*MEAS_COND_AXIS_OFFSET-45 "0"
*MEAS_COND_AXIS_OFFSET-46 "0"
*MEAS_COND_AXIS_OFFSET-47 "0"
*MEAS_COND_AXIS_OFFSET-48 "0"
*MEAS_COND_AXIS_OFFSET-49 "0"
*MEAS_COND_AXIS_OFFSET-5 "0"
*MEAS_COND_AXIS_OFFSET-50 "0"
*MEAS_COND_AXIS_OFFSET-51 "0"
*MEAS_COND_AXIS_OFFSET-52 "0"
*MEAS_COND_AXIS_OFFSET-53 "0"
*MEAS_COND_AXIS_OFFSET-54 "0"
*MEAS_COND_AXIS_OFFSET-55 "0"
*MEAS_COND_AXIS_OFFSET-56 "0"
*MEAS_COND_AXIS_OFFSET-57 "0"
*MEAS_COND_AXIS_OFFSET-58 "0"
*MEAS_COND_AXIS_OFFSET-59 "0"
*MEAS_COND_AXIS_OFFSET-6 "0"
*MEAS_COND_AXIS_OFFSET-60 "0"
*MEAS_COND_AXIS_OFFSET-61 "0"
*MEAS_COND_AXIS_OFFSET-62 "0"
*MEAS_COND_AXIS_OFFSET-63 "0"
*MEAS_COND_AXIS_OFFSET-64 "0"
*MEAS_COND_AXIS_OFFSET-65 "0"
*MEAS_COND_AXIS_OFFSET-66 "0"
*MEAS_COND_AXIS_OFFSET-67 "0"
*MEAS_COND_AXIS_OFFSET-68 "0"
*MEAS_COND_AXIS_OFFSET-69 "0"
*MEAS_COND_AXIS_OFFSET-7 "0"
*MEAS_COND_AXIS_OFFSET-70 "0"
*MEAS_COND_AXIS_OFFSET-71 "0"
*MEAS_COND_AXIS_OFFSET-72 "0"
*MEAS_COND_AXIS_OFFSET-73 "0"
*MEAS_COND_AXIS_OFFSET-74 "0"
*MEAS_COND_AXIS_OFFSET-75 "0"
*MEAS_COND_AXIS_OFFSET-76 "0"
*MEAS_COND_AXIS_OFFSET-77 "0"
*MEAS_COND_AXIS_OFFSET-78 "0"
*MEAS_COND_AXIS_OFFSET-79 "0"
*MEAS_COND_AXIS_OFFSET-8 "0"
*MEAS_COND_AXIS_OFFSET-80 "0"
*MEAS_COND_AXIS_OFFSET-81 "0"
*MEAS_COND_AXIS_OFFSET-82 "0"
*MEAS_COND_AXIS_OFFSET-83 "0"
*MEAS_COND_AXIS_OFFSET-84 "0"
*MEAS_COND_AXIS_OFFSET-85 "0"
*MEAS_COND_AXIS_OFFSET-86 "0"
*MEAS_COND_AXIS_OFFSET-87 "0"
*MEAS_COND_AXIS_OFFSET-88 "0"
*MEAS_COND_AXIS_OFFSET-89 "0"
*MEAS_COND_AXIS_OFFSET-9 "0"
*MEAS_COND_AXIS_OFFSET-90 "0"
*MEAS_COND_AXIS_OFFSET-91 "0"
*MEAS_COND_AXIS_OFFSET-92 "0"
*MEAS_COND_AXIS_OFFSET-93 "0"
*MEAS_COND_AXIS_OFFSET-94 "0"
*MEAS_COND_AXIS_OFFSET-95 "0"
*MEAS_COND_AXIS_OFFSET-96 "0"
*MEAS_COND_AXIS_OFFSET-97 "0"
*MEAS_COND_AXIS_OFFSET-98 "0"
*MEAS_COND_AXIS_OFFSET-99 "0"
*MEAS_COND_AXIS_POSITION-0 "0.0000"
*MEAS_COND_AXIS_POSITION-1 "0.0000"
*MEAS_COND_AXIS_POSITION-10 "Right"
*MEAS_COND_AXIS_POSITION-100 "IPS_adaptor"
*MEAS_COND_AXIS_POSITION-101 "PSA_open"
*MEAS_COND_AXIS_POSITION-102 "None"
*MEAS_COND_AXIS_POSITION-11 "None"
*MEAS_COND_AXIS_POSITION-12 "Type2"
*MEAS_COND_AXIS_POSITION-13 "0"
*MEAS_COND_AXIS_POSITION-14 "BB"
*MEAS_COND_AXIS_POSITION-15 "Soller_slit_2.5deg"
*MEAS_COND_AXIS_POSITION-16 "1/2deg"
*MEAS_COND_AXIS_POSITION-17 "1.514375"
*MEAS_COND_AXIS_POSITION-18 "{meta_dict.get('Divergence Slit (DS)', '1/2deg')}"
*MEAS_COND_AXIS_POSITION-19 "-0.0393750"
*MEAS_COND_AXIS_POSITION-2 "0.0000"
*MEAS_COND_AXIS_POSITION-20 "2mm"
*MEAS_COND_AXIS_POSITION-21 "2mm"
*MEAS_COND_AXIS_POSITION-22 "2mm"
*MEAS_COND_AXIS_POSITION-23 "None"
*MEAS_COND_AXIS_POSITION-24 "0.000"
*MEAS_COND_AXIS_POSITION-25 "0.000"
*MEAS_COND_AXIS_POSITION-26 "0.5094"
*MEAS_COND_AXIS_POSITION-27 "90.000"
*MEAS_COND_AXIS_POSITION-28 "0.000"
*MEAS_COND_AXIS_POSITION-29 "0.000"
*MEAS_COND_AXIS_POSITION-3 "0.0000"
*MEAS_COND_AXIS_POSITION-30 "90.00"
*MEAS_COND_AXIS_POSITION-31 "0.000"
*MEAS_COND_AXIS_POSITION-32 "0.0000"
*MEAS_COND_AXIS_POSITION-33 "Open"
*MEAS_COND_AXIS_POSITION-34 "20.000000"
*MEAS_COND_AXIS_POSITION-35 "{meta_dict.get('Scattering Slit (SS)', 'Open')}"
*MEAS_COND_AXIS_POSITION-36 "-0.2990625"
*MEAS_COND_AXIS_POSITION-37 "Cu_K-beta_1D"
*MEAS_COND_AXIS_POSITION-38 "PSA_open"
*MEAS_COND_AXIS_POSITION-39 "Soller_slit_2.5deg"
*MEAS_COND_AXIS_POSITION-4 "0.0000"
*MEAS_COND_AXIS_POSITION-40 "20.100mm"
*MEAS_COND_AXIS_POSITION-41 "20.1"
*MEAS_COND_AXIS_POSITION-42 "{meta_dict.get('Receiving Slit (RS)', '20.100mm')}"
*MEAS_COND_AXIS_POSITION-43 "AC"
*MEAS_COND_AXIS_POSITION-44 "Mo-HiC"
*MEAS_COND_AXIS_POSITION-46 "None"
*MEAS_COND_AXIS_POSITION-47 "None"
*MEAS_COND_AXIS_POSITION-48 "None"
*MEAS_COND_AXIS_POSITION-5 "0.0000"
*MEAS_COND_AXIS_POSITION-6 "0.0000"
*MEAS_COND_AXIS_POSITION-7 "0.0000"
*MEAS_COND_AXIS_POSITION-8 "Middle"
*MEAS_COND_AXIS_POSITION-9 "-4.0000000"
*MEAS_COND_AXIS_POSITION-98 "1"
*MEAS_COND_AXIS_POSITION-99 "Standard"
*MEAS_COND_AXIS_RESOLUTION-0 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-1 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-10 ""
*MEAS_COND_AXIS_RESOLUTION-11 ""
*MEAS_COND_AXIS_RESOLUTION-12 ""
*MEAS_COND_AXIS_RESOLUTION-13 "1"
*MEAS_COND_AXIS_RESOLUTION-14 ""
*MEAS_COND_AXIS_RESOLUTION-15 ""
*MEAS_COND_AXIS_RESOLUTION-16 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-17 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-18 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-19 "0.0003125"
*MEAS_COND_AXIS_RESOLUTION-2 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-20 ""
*MEAS_COND_AXIS_RESOLUTION-21 ""
*MEAS_COND_AXIS_RESOLUTION-22 ""
*MEAS_COND_AXIS_RESOLUTION-23 ""
*MEAS_COND_AXIS_RESOLUTION-24 "0.001"
*MEAS_COND_AXIS_RESOLUTION-25 "0.002"
*MEAS_COND_AXIS_RESOLUTION-26 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-27 "0.001"
*MEAS_COND_AXIS_RESOLUTION-28 "0.002"
*MEAS_COND_AXIS_RESOLUTION-29 "0.004"
*MEAS_COND_AXIS_RESOLUTION-3 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-30 "0.01"
*MEAS_COND_AXIS_RESOLUTION-31 "0.002"
*MEAS_COND_AXIS_RESOLUTION-32 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-33 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-34 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-35 "1/1600"
*MEAS_COND_AXIS_RESOLUTION-36 "0.0003125"
*MEAS_COND_AXIS_RESOLUTION-37 ""
*MEAS_COND_AXIS_RESOLUTION-38 ""
*MEAS_COND_AXIS_RESOLUTION-39 ""
*MEAS_COND_AXIS_RESOLUTION-4 "0.0002"
*MEAS_COND_AXIS_RESOLUTION-40 "0.1"
*MEAS_COND_AXIS_RESOLUTION-41 "0.1"
*MEAS_COND_AXIS_RESOLUTION-42 "0.1"
*MEAS_COND_AXIS_RESOLUTION-43 ""
*MEAS_COND_AXIS_RESOLUTION-44 ""
*MEAS_COND_AXIS_RESOLUTION-45 "1"
*MEAS_COND_AXIS_RESOLUTION-46 ""
*MEAS_COND_AXIS_RESOLUTION-47 ""
*MEAS_COND_AXIS_RESOLUTION-48 ""
*MEAS_COND_AXIS_RESOLUTION-49 "0.01"
*MEAS_COND_AXIS_RESOLUTION-5 "0.0002"
*MEAS_COND_AXIS_RESOLUTION-50 "0.010000"
*MEAS_COND_AXIS_RESOLUTION-51 "0.01"
*MEAS_COND_AXIS_RESOLUTION-52 "0.01"
*MEAS_COND_AXIS_RESOLUTION-53 "0.01"
*MEAS_COND_AXIS_RESOLUTION-54 "1"
*MEAS_COND_AXIS_RESOLUTION-55 "1"
*MEAS_COND_AXIS_RESOLUTION-56 "1"
*MEAS_COND_AXIS_RESOLUTION-57 "1"
*MEAS_COND_AXIS_RESOLUTION-58 "0.01"
*MEAS_COND_AXIS_RESOLUTION-59 "0.01"
*MEAS_COND_AXIS_RESOLUTION-6 "0.0001"
*MEAS_COND_AXIS_RESOLUTION-60 "0.01"
*MEAS_COND_AXIS_RESOLUTION-61 "0.01"
*MEAS_COND_AXIS_RESOLUTION-62 "0.010000"
*MEAS_COND_AXIS_RESOLUTION-63 "1"
*MEAS_COND_AXIS_RESOLUTION-64 "0.100000"
*MEAS_COND_AXIS_RESOLUTION-65 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-66 "0.010000"
*MEAS_COND_AXIS_RESOLUTION-67 "0.01"
*MEAS_COND_AXIS_RESOLUTION-68 "0.01"
*MEAS_COND_AXIS_RESOLUTION-69 "0.1"
*MEAS_COND_AXIS_RESOLUTION-7 "0.0005"
*MEAS_COND_AXIS_RESOLUTION-70 "1"
*MEAS_COND_AXIS_RESOLUTION-71 "0.01"
*MEAS_COND_AXIS_RESOLUTION-72 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-73 "1"
*MEAS_COND_AXIS_RESOLUTION-74 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-75 "1"
*MEAS_COND_AXIS_RESOLUTION-76 "0.01"
*MEAS_COND_AXIS_RESOLUTION-77 "1"
*MEAS_COND_AXIS_RESOLUTION-78 "1"
*MEAS_COND_AXIS_RESOLUTION-79 "1"
*MEAS_COND_AXIS_RESOLUTION-8 ""
*MEAS_COND_AXIS_RESOLUTION-80 "0.01"
*MEAS_COND_AXIS_RESOLUTION-81 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-82 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-83 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-84 "-"
*MEAS_COND_AXIS_RESOLUTION-85 "-"
*MEAS_COND_AXIS_RESOLUTION-86 "-"
*MEAS_COND_AXIS_RESOLUTION-87 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-88 "-"
*MEAS_COND_AXIS_RESOLUTION-89 "-"
*MEAS_COND_AXIS_RESOLUTION-9 "0.0000625"
*MEAS_COND_AXIS_RESOLUTION-90 "-"
*MEAS_COND_AXIS_RESOLUTION-91 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-92 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-93 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-94 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-95 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-96 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-97 "1.000000"
*MEAS_COND_AXIS_RESOLUTION-98 ""
*MEAS_COND_AXIS_STATE-0 "Fixed"
*MEAS_COND_AXIS_STATE-1 "Fixed"
*MEAS_COND_AXIS_STATE-10 "Fixed"
*MEAS_COND_AXIS_STATE-11 "Fixed"
*MEAS_COND_AXIS_STATE-12 "Fixed"
*MEAS_COND_AXIS_STATE-13 "Fixed"
*MEAS_COND_AXIS_STATE-14 "Fixed"
*MEAS_COND_AXIS_STATE-15 "Fixed"
*MEAS_COND_AXIS_STATE-16 "Fixed"
*MEAS_COND_AXIS_STATE-17 "Fixed"
*MEAS_COND_AXIS_STATE-18 "Fixed"
*MEAS_COND_AXIS_STATE-19 "Fixed"
*MEAS_COND_AXIS_STATE-2 "Fixed"
*MEAS_COND_AXIS_STATE-20 "Fixed"
*MEAS_COND_AXIS_STATE-21 "Fixed"
*MEAS_COND_AXIS_STATE-22 "Fixed"
*MEAS_COND_AXIS_STATE-23 "Fixed"
*MEAS_COND_AXIS_STATE-24 "Fixed"
*MEAS_COND_AXIS_STATE-25 "Fixed"
*MEAS_COND_AXIS_STATE-26 "Fixed"
*MEAS_COND_AXIS_STATE-27 "Fixed"
*MEAS_COND_AXIS_STATE-28 "Fixed"
*MEAS_COND_AXIS_STATE-29 "Fixed"
*MEAS_COND_AXIS_STATE-3 "Fixed"
*MEAS_COND_AXIS_STATE-30 "Fixed"
*MEAS_COND_AXIS_STATE-31 "Fixed"
*MEAS_COND_AXIS_STATE-32 "Fixed"
*MEAS_COND_AXIS_STATE-33 "Fixed"
*MEAS_COND_AXIS_STATE-34 "Fixed"
*MEAS_COND_AXIS_STATE-35 "Fixed"
*MEAS_COND_AXIS_STATE-36 "Fixed"
*MEAS_COND_AXIS_STATE-37 "Fixed"
*MEAS_COND_AXIS_STATE-38 "Fixed"
*MEAS_COND_AXIS_STATE-39 "Fixed"
*MEAS_COND_AXIS_STATE-4 "Scan"
*MEAS_COND_AXIS_STATE-40 "Fixed"
*MEAS_COND_AXIS_STATE-41 "Fixed"
*MEAS_COND_AXIS_STATE-42 "Fixed"
*MEAS_COND_AXIS_STATE-43 "Fixed"
*MEAS_COND_AXIS_STATE-44 "Fixed"
*MEAS_COND_AXIS_STATE-45 "Fixed"
*MEAS_COND_AXIS_STATE-46 "Fixed"
*MEAS_COND_AXIS_STATE-47 "Fixed"
*MEAS_COND_AXIS_STATE-48 "Fixed"
*MEAS_COND_AXIS_STATE-49 "Fixed"
*MEAS_COND_AXIS_STATE-5 "Fixed"
*MEAS_COND_AXIS_STATE-50 "Fixed"
*MEAS_COND_AXIS_STATE-51 "Fixed"
*MEAS_COND_AXIS_STATE-52 "Fixed"
*MEAS_COND_AXIS_STATE-53 "Fixed"
*MEAS_COND_AXIS_STATE-54 "Fixed"
*MEAS_COND_AXIS_STATE-55 "Fixed"
*MEAS_COND_AXIS_STATE-56 "Fixed"
*MEAS_COND_AXIS_STATE-57 "Fixed"
*MEAS_COND_AXIS_STATE-58 "Fixed"
*MEAS_COND_AXIS_STATE-59 "Fixed"
*MEAS_COND_AXIS_STATE-6 "Fixed"
*MEAS_COND_AXIS_STATE-60 "Fixed"
*MEAS_COND_AXIS_STATE-61 "Fixed"
*MEAS_COND_AXIS_STATE-62 "Fixed"
*MEAS_COND_AXIS_STATE-63 "Fixed"
*MEAS_COND_AXIS_STATE-64 "Fixed"
*MEAS_COND_AXIS_STATE-65 "Fixed"
*MEAS_COND_AXIS_STATE-66 "Fixed"
*MEAS_COND_AXIS_STATE-67 "Fixed"
*MEAS_COND_AXIS_STATE-68 "Fixed"
*MEAS_COND_AXIS_STATE-69 "Fixed"
*MEAS_COND_AXIS_STATE-7 "Fixed"
*MEAS_COND_AXIS_STATE-70 "Fixed"
*MEAS_COND_AXIS_STATE-71 "Fixed"
*MEAS_COND_AXIS_STATE-72 "Fixed"
*MEAS_COND_AXIS_STATE-73 "Fixed"
*MEAS_COND_AXIS_STATE-74 "Fixed"
*MEAS_COND_AXIS_STATE-75 "Fixed"
*MEAS_COND_AXIS_STATE-76 "Fixed"
*MEAS_COND_AXIS_STATE-77 "Fixed"
*MEAS_COND_AXIS_STATE-78 "Fixed"
*MEAS_COND_AXIS_STATE-79 "Fixed"
*MEAS_COND_AXIS_STATE-8 "Fixed"
*MEAS_COND_AXIS_STATE-80 "Fixed"
*MEAS_COND_AXIS_STATE-81 "Fixed"
*MEAS_COND_AXIS_STATE-82 "Fixed"
*MEAS_COND_AXIS_STATE-83 "Fixed"
*MEAS_COND_AXIS_STATE-84 "Fixed"
*MEAS_COND_AXIS_STATE-85 "Fixed"
*MEAS_COND_AXIS_STATE-86 "Fixed"
*MEAS_COND_AXIS_STATE-87 "Fixed"
*MEAS_COND_AXIS_STATE-88 "Fixed"
*MEAS_COND_AXIS_STATE-89 "Fixed"
*MEAS_COND_AXIS_STATE-9 "Fixed"
*MEAS_COND_AXIS_STATE-90 "Fixed"
*MEAS_COND_AXIS_STATE-91 "Fixed"
*MEAS_COND_AXIS_STATE-92 "Fixed"
*MEAS_COND_AXIS_STATE-93 "Fixed"
*MEAS_COND_AXIS_STATE-94 "Fixed"
*MEAS_COND_AXIS_STATE-95 "Fixed"
*MEAS_COND_AXIS_STATE-96 "Fixed"
*MEAS_COND_AXIS_STATE-97 "Fixed"
*MEAS_COND_AXIS_STATE-98 "Fixed"
*MEAS_COND_AXIS_UNIT-0 "deg"
*MEAS_COND_AXIS_UNIT-1 "deg"
*MEAS_COND_AXIS_UNIT-10 ""
*MEAS_COND_AXIS_UNIT-100 ""
*MEAS_COND_AXIS_UNIT-101 ""
*MEAS_COND_AXIS_UNIT-102 ""
*MEAS_COND_AXIS_UNIT-11 ""
*MEAS_COND_AXIS_UNIT-12 ""
*MEAS_COND_AXIS_UNIT-13 "pulse"
*MEAS_COND_AXIS_UNIT-14 ""
*MEAS_COND_AXIS_UNIT-15 ""
*MEAS_COND_AXIS_UNIT-16 ""
*MEAS_COND_AXIS_UNIT-17 "mm"
*MEAS_COND_AXIS_UNIT-18 ""
*MEAS_COND_AXIS_UNIT-19 "mm"
*MEAS_COND_AXIS_UNIT-2 "deg"
*MEAS_COND_AXIS_UNIT-20 ""
*MEAS_COND_AXIS_UNIT-21 ""
*MEAS_COND_AXIS_UNIT-22 ""
*MEAS_COND_AXIS_UNIT-23 ""
*MEAS_COND_AXIS_UNIT-24 "deg"
*MEAS_COND_AXIS_UNIT-25 "deg"
*MEAS_COND_AXIS_UNIT-26 "mm"
*MEAS_COND_AXIS_UNIT-27 "deg"
*MEAS_COND_AXIS_UNIT-28 "deg"
*MEAS_COND_AXIS_UNIT-29 "deg"
*MEAS_COND_AXIS_UNIT-3 "deg"
*MEAS_COND_AXIS_UNIT-30 "deg"
*MEAS_COND_AXIS_UNIT-31 "deg"
*MEAS_COND_AXIS_UNIT-32 "deg"
*MEAS_COND_AXIS_UNIT-33 ""
*MEAS_COND_AXIS_UNIT-34 "mm"
*MEAS_COND_AXIS_UNIT-35 ""
*MEAS_COND_AXIS_UNIT-36 "mm"
*MEAS_COND_AXIS_UNIT-37 ""
*MEAS_COND_AXIS_UNIT-38 ""
*MEAS_COND_AXIS_UNIT-39 ""
*MEAS_COND_AXIS_UNIT-4 "deg"
*MEAS_COND_AXIS_UNIT-40 ""
*MEAS_COND_AXIS_UNIT-41 "mm"
*MEAS_COND_AXIS_UNIT-42 ""
*MEAS_COND_AXIS_UNIT-43 ""
*MEAS_COND_AXIS_UNIT-44 ""
*MEAS_COND_AXIS_UNIT-45 "keV"
*MEAS_COND_AXIS_UNIT-46 ""
*MEAS_COND_AXIS_UNIT-47 ""
*MEAS_COND_AXIS_UNIT-48 ""
*MEAS_COND_AXIS_UNIT-49 "H"
*MEAS_COND_AXIS_UNIT-5 "deg"
*MEAS_COND_AXIS_UNIT-50 "H"
*MEAS_COND_AXIS_UNIT-51 "H"
*MEAS_COND_AXIS_UNIT-52 "H"
*MEAS_COND_AXIS_UNIT-53 "H"
*MEAS_COND_AXIS_UNIT-54 "mV"
*MEAS_COND_AXIS_UNIT-55 "V"
*MEAS_COND_AXIS_UNIT-56 "V"
*MEAS_COND_AXIS_UNIT-57 "V"
*MEAS_COND_AXIS_UNIT-58 "H"
*MEAS_COND_AXIS_UNIT-59 "H"
*MEAS_COND_AXIS_UNIT-6 "deg"
*MEAS_COND_AXIS_UNIT-60 "H"
*MEAS_COND_AXIS_UNIT-61 "H"
*MEAS_COND_AXIS_UNIT-62 "H"
*MEAS_COND_AXIS_UNIT-63 "uS/m"
*MEAS_COND_AXIS_UNIT-64 "L/min"
*MEAS_COND_AXIS_UNIT-65 "degree"
*MEAS_COND_AXIS_UNIT-66 "MPa"
*MEAS_COND_AXIS_UNIT-67 "MPa"
*MEAS_COND_AXIS_UNIT-68 "MPa"
*MEAS_COND_AXIS_UNIT-69 "L/min"
*MEAS_COND_AXIS_UNIT-7 "deg"
*MEAS_COND_AXIS_UNIT-70 "C"
*MEAS_COND_AXIS_UNIT-71 "MPa"
*MEAS_COND_AXIS_UNIT-72 "degree"
*MEAS_COND_AXIS_UNIT-73 "percent"
*MEAS_COND_AXIS_UNIT-74 "degree"
*MEAS_COND_AXIS_UNIT-75 "C"
*MEAS_COND_AXIS_UNIT-76 "H"
*MEAS_COND_AXIS_UNIT-77 "time"
*MEAS_COND_AXIS_UNIT-78 "msec"
*MEAS_COND_AXIS_UNIT-79 "msec"
*MEAS_COND_AXIS_UNIT-8 ""
*MEAS_COND_AXIS_UNIT-80 "H"
*MEAS_COND_AXIS_UNIT-81 "times"
*MEAS_COND_AXIS_UNIT-82 "msec"
*MEAS_COND_AXIS_UNIT-83 "msec"
*MEAS_COND_AXIS_UNIT-84 "-"
*MEAS_COND_AXIS_UNIT-85 "-"
*MEAS_COND_AXIS_UNIT-86 "-"
*MEAS_COND_AXIS_UNIT-87 ""
*MEAS_COND_AXIS_UNIT-88 "-"
*MEAS_COND_AXIS_UNIT-89 "-"
*MEAS_COND_AXIS_UNIT-9 "mm"
*MEAS_COND_AXIS_UNIT-90 "-"
*MEAS_COND_AXIS_UNIT-91 ""
*MEAS_COND_AXIS_UNIT-92 ""
*MEAS_COND_AXIS_UNIT-93 ""
*MEAS_COND_AXIS_UNIT-94 ""
*MEAS_COND_AXIS_UNIT-95 ""
*MEAS_COND_AXIS_UNIT-96 ""
*MEAS_COND_AXIS_UNIT-97 ""
*MEAS_COND_AXIS_UNIT-98 ""
*MEAS_COND_AXIS_UNIT-99 ""
*MEAS_COND_COUNTER_CENTER_X "382.5"
*MEAS_COND_COUNTER_CENTER_Y "197.5"
*MEAS_COND_COUNTER_COUNTMODE "Differential"
*MEAS_COND_COUNTER_DEADTIMECORRECTION "Enabled"
*MEAS_COND_COUNTER_DISTANCE "300"
*MEAS_COND_COUNTER_ENERGYMODE "Standard"
*MEAS_COND_COUNTER_INTEGRALMODE "Line"
*MEAS_COND_COUNTER_PHA_UNIT "keV"
*MEAS_COND_COUNTER_PHABASE "6.0"
*MEAS_COND_COUNTER_PHAWINDOW "4.0"
*MEAS_COND_COUNTER_PITCH_X "0.1"
*MEAS_COND_COUNTER_PITCH_Y "0.1"
*MEAS_COND_COUNTER_PITCHUNIT "mm"
*MEAS_COND_COUNTER_VALIDWIDTH_X "200"
*MEAS_COND_COUNTER_VALIDWIDTH_Y "201"
*MEAS_COND_OPT_ATTR "BB"
*MEAS_COND_OPT_NAME "User defined settings"
*MEAS_COND_XG_CURRENT "{meta_dict.get('MEAS_COND_XG_CURRENT', '30')}"
*MEAS_COND_XG_VOLTAGE "{meta_dict.get('MEAS_COND_XG_VOLTAGE', '40')}"
*MEAS_COND_XG_WAVE_TYPE "Ka"
*MEAS_DATA_COUNT "{data_count}"
*MEAS_SCAN_AXIS_X "{meta_dict.get('MEAS_SCAN_AXIS_X', 'TwoThetaTheta')}"
*MEAS_SCAN_AXIS_X_INTERNAL "TwoThetaTheta"
*MEAS_SCAN_END_TIME "{formatted_end_time}"
*MEAS_SCAN_MODE "{meta_dict.get('MEAS_SCAN_MODE', 'CONTINUOUS')}"
*MEAS_SCAN_MODE_INTERNAL "TDI_1D"
*MEAS_SCAN_RESOLUTION_X "0.0002"
*MEAS_SCAN_SPEED "{meta_dict.get('MEAS_SCAN_SPEED', '5.0000')}"
*MEAS_SCAN_SPEED_UNIT "{meta_dict.get('MEAS_SCAN_SPEED_UNIT', 'deg/min')}"
*MEAS_SCAN_START "{start_angle}"
*MEAS_SCAN_START_TIME "{formatted_start_time}"
*MEAS_SCAN_STEP "{step_size}"
*MEAS_SCAN_STOP "{stop_angle}"
*MEAS_SCAN_UNEQUALY_SPACED "False"
*MEAS_SCAN_UNIT_X "deg"
*MEAS_SCAN_UNIT_Y "counts"
*RAS_HEADER_END"""

        data_lines = []
        for _, row in data_df.iterrows():
            data_lines.append(f"{row['2Theta']:.6f} {row['Intensity']:.4f} 1.0000")

        return "\n".join([
            header_template,
            "*RAS_INT_START",
            *data_lines,
            "*RAS_INT_END",
            "*RAS_DATA_END",
            "*DSC_DATA_END"
        ])

    def generate_raw(metadata_df, data_df):
        meta_dict = pd.Series(metadata_df.Value.values, index=metadata_df.Parameter).to_dict()
        header = bytearray(2600)

        try:
            start_angle = float(meta_dict.get('Start Angle (°)', data_df['2Theta'].min()))
            num_points = len(data_df)
            step_size = float(meta_dict.get('Step Size (°)', (data_df['2Theta'].iloc[1] - data_df['2Theta'].iloc[0])))
            time_per_step = float(meta_dict.get('Time per Step (s)', 1.0))
            k_alpha1 = float(meta_dict.get('K-Alpha1 (Å)', 1.54060))
            k_alpha2 = float(meta_dict.get('K-Alpha2 (Å)', 1.54439))
            k_beta = float(meta_dict.get('K-Beta (Å)', 1.39225))
            target_name = meta_dict.get('X-ray Target', 'Cu').ljust(12).encode('utf-8')

            struct.pack_into('<f', header, 136, start_angle)
            struct.pack_into('<f', header, 140, step_size)
            struct.pack_into('<i', header, 148, num_points)
            struct.pack_into('<f', header, 152, time_per_step)
            struct.pack_into('12s', header, 244, target_name)
            struct.pack_into('<f', header, 308, k_alpha1)
            struct.pack_into('<f', header, 312, k_alpha2)
            struct.pack_into('<f', header, 316, k_beta)
            date_str = datetime.now().strftime("%d-%b-%Y").ljust(9).encode('utf-8')
            struct.pack_into('9s', header, 38, date_str)
        except Exception as e:
            st.error(f"Error preparing RAW file header: {e}")
            return None

        intensities = data_df['Intensity'].values.astype(np.float32)
        data_bytes = intensities.tobytes()
        return bytes(header) + data_bytes

    def get_default_metadata(format_type='XRDML'):
        now = datetime.now()
        if format_type == 'RAS':
            metadata = {
                'FILE_OPERATOR': 'XRDlicious User', 'HW_XG_TARGET_NAME': 'Cu',
                'MEAS_COND_XG_VOLTAGE': '40', 'MEAS_COND_XG_CURRENT': '30',
                'MEAS_SCAN_AXIS_X': 'TwoThetaTheta', 'MEAS_SCAN_MODE': 'CONTINUOUS',
                'MEAS_SCAN_SPEED': '5.0000', 'MEAS_SCAN_SPEED_UNIT': 'deg/min',
                'Divergence Slit (DS)': '1/2deg', 'Scattering Slit (SS)': 'Open',
                'Receiving Slit (RS)': '20.100mm',
                'MEAS_SCAN_START_TIME': now.isoformat(),
                'MEAS_SCAN_END_TIME': (now + timedelta(minutes=20)).isoformat(),
            }
        elif format_type == 'RAW':
            metadata = {
                'X-ray Target': 'Cu', 'Time per Step (s)': '1.0',
                'K-Alpha1 (Å)': '1.54060', 'K-Alpha2 (Å)': '1.54439',
                'K-Beta (Å)': '1.39225',
            }
        else:  # XRDML
            metadata = {
                'Sample ID': '000000-0000 - Converted with XRDlicious', 'Sample Name': 'Converted Sample Name',
                'Status': 'Completed', 'Measurement Type': 'Scan', 'Scan Mode': 'Continuous',
                'Scan Axis': 'Gonio', 'Start Time': now.isoformat(),
                'End Time': (now + timedelta(minutes=30)).isoformat(),
                'Author': 'XRDlicious User', 'Application Software': 'XRDlicious Converter',
                'K-Alpha1 Wavelength (Å)': '1.54056', 'K-Alpha2 Wavelength (Å)': '1.54439',
                'K-Beta Wavelength (Å)': '1.39225', 'Ratio K-Alpha2/K-Alpha1': '0.5',
                'Common Counting Time (s)': '1.0', 'X-ray Tube Tension (kV)': '45',
                'X-ray Tube Current (mA)': '40', 'Anode Material': 'Cu',
            }
        return pd.DataFrame(list(metadata.items()), columns=['Parameter', 'Value'])

    st.markdown("### 📜 XRD File Format Converter (.xrdml, .ras, .raw, .xy)")
    #st.markdown(
    #    """
    #    <div style="background-color:#f8d7da; padding:6px 10px; border-radius:4px; border:1px solid #f5c2c7; width: fit-content;">
    #        <span style="color:#842029; font-size:14px;">🔧 Testing mode</span>
    #    </div>
    #    """,
    #    unsafe_allow_html=True
    #)
    st.info(
        "📄🔁📄 Upload one or more data powder diffraction files to convert them to a different format. .**xy ➡️ .xrdml, .ras, .raw**. "
        "Or **.xrdml, .ras, .raw ➡️ .xy**. \n\n ⚠️ Older **.raw** format can currently produce incorrect x-axis values. "
        "Check if they are correct in the converted .xy format.")

    allow_batch = st.checkbox(
        f"Allow multiple file uploads (**batch mode**). All files must have the same format. Plot from the first file will be previewed. The set settings "
        f"will propagated to all converted files." ,
    )


    uploaded_files_raw = st.file_uploader("Upload Data File(s)",
                                          type=["xrdml", "xml", "ras", "xy", "dat", "txt", "raw"],
                                          accept_multiple_files=allow_batch)

    if uploaded_files_raw:
        if isinstance(uploaded_files_raw, list):
            uploaded_files = uploaded_files_raw
        else:
            uploaded_files = [uploaded_files_raw]

        first_file_ext = uploaded_files[0].name.lower().split('.')[-1]
        if not all(f.name.lower().split('.')[-1] == first_file_ext for f in uploaded_files):
            st.error("Error: Please upload files of the same format.")
            return

        first_file = uploaded_files[0]
        file_ext = first_file_ext
        data_df = None
        is_batch = len(uploaded_files) > 1
        if file_ext in ['xrdml', 'xml', 'ras', 'rasx', 'raw']:
            key_metadata = {}
            full_metadata = {}
            if file_ext == 'raw':
                file_content_bytes = first_file.getvalue()
                key_metadata, data_df = parse_raw(file_content_bytes)
                full_metadata = key_metadata
            elif file_ext in ['xrdml', 'xml', 'ras', 'rasx']:
                if file_ext == 'ras':
                    file_content = first_file.getvalue().decode("utf-8", errors='replace')
                    full_metadata, key_metadata, data_df = parse_ras(file_content)
                elif file_ext == 'rasx':
                    full_metadata, key_metadata, data_df = parse_rasx(first_file)
                else:
                    file_content = first_file.getvalue().decode("utf-8", errors='replace')
                    key_metadata, data_df = parse_xrdml(file_content)
                    full_metadata = key_metadata

            if data_df is not None:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.markdown(f"#### 📝 Key Parameters (from `{first_file.name}`)")
                    st.table(pd.DataFrame(list(key_metadata.items()), columns=['Parameter', 'Value']))

                    if full_metadata and file_ext in ['ras', 'rasx']:
                        with st.expander("Show Full Raw Header"):
                            st.dataframe(pd.DataFrame(list(full_metadata.items()), columns=['Parameter', 'Value']),
                                         height=300)

                    include_header = st.checkbox("Include header in .xy file", value=False)

                    if is_batch:
                        st.write(f"**Batch conversion for {len(uploaded_files)} files.**")
                        if st.button("⬇️ Download All as .xy (.zip)", type="primary", use_container_width=True):
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for uploaded_file in uploaded_files:
                                    df_to_convert = None
                                    if file_ext == 'raw':
                                        _, df_to_convert = parse_raw(uploaded_file.getvalue())
                                    elif file_ext == 'ras':
                                        _, _, df_to_convert = parse_ras(
                                            uploaded_file.getvalue().decode("utf-8", errors='replace'))
                                    elif file_ext == 'rasx':
                                        _, _, df_to_convert = parse_rasx(uploaded_file)
                                    else:
                                        _, df_to_convert = parse_xrdml(
                                            uploaded_file.getvalue().decode("utf-8", errors='replace'))

                                    if df_to_convert is not None:
                                        new_filename = uploaded_file.name.rsplit('.', 1)[0] + '.xy'
                                        xy_data = convert_to_xy(df_to_convert, include_header)
                                        zf.writestr(new_filename, xy_data)

                            st.download_button(
                                label="📦 Download ZIP",
                                data=zip_buffer.getvalue(),
                                file_name="converted_xy_files.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                    else:
                        default_name = first_file.name.rsplit('.', 1)[0] + '.xy'
                        download_filename = st.text_input("Enter filename for download:", default_name)
                        xy_data = convert_to_xy(data_df, include_header)
                        st.download_button("⬇️ Download as .xy File", xy_data, download_filename, "text/plain",
                                           type="primary", use_container_width=True)

                with col2:
                    st.markdown("#### 📈 Diffraction Pattern")
                    fig = go.Figure(
                        go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='lines', name='Intensity'))
                    fig.update_layout(title=f"Data from {first_file.name}", xaxis_title="2θ (°)",
                                      yaxis_title="Intensity (counts)", height=550, margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)


        elif file_ext in ['xy', 'dat', 'txt']:
            file_content = first_file.getvalue().decode("utf-8", errors='replace')
            data_df = parse_xy(file_content)

            if data_df is not None:
                col1, col2 = st.columns([1, 1.5])
                with col1:
                    st.markdown("#### 📝 Edit Details for Output File(s)")
                    if is_batch:
                        st.info(f"These settings will be applied to all **{len(uploaded_files)} files**.")
                    output_format = st.selectbox("Select Output Format", ['XRDML', 'RAS', 'RAW'])

                    df_state_key = f"meta_df_{output_format}_{first_file.name}"
                    if st.session_state.get('last_file_format_choice') != (first_file.name, output_format):
                        st.session_state[df_state_key] = get_default_metadata(output_format)
                        st.session_state['last_file_format_choice'] = (first_file.name, output_format)

                    edited_df = st.data_editor(st.session_state[df_state_key], num_rows="dynamic", height=425,
                                               key=f"editor_{output_format}")

                    if st.button("Apply Changes & Prepare Download", use_container_width=True):
                        st.session_state[df_state_key] = edited_df
                        st.success(f"Settings applied. {output_format} file is ready for download below.")

                    file_extensions = {'XRDML': 'xrdml', 'RAS': 'ras', 'RAW': 'raw'}
                    file_extension = file_extensions.get(output_format, 'txt')

                    if df_state_key in st.session_state:
                        applied_metadata_df = st.session_state[df_state_key]
                        if is_batch:
                            zip_buffer = BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                                for uploaded_file in uploaded_files:
                                    df_to_convert = parse_xy(uploaded_file.getvalue().decode("utf-8", errors='replace'))
                                    if df_to_convert is not None:
                                        new_filename = uploaded_file.name.rsplit('.', 1)[0] + f'.{file_extension}'
                                        file_content_to_download = None
                                        if output_format == 'RAS':
                                            file_content_to_download = generate_ras(applied_metadata_df, df_to_convert)
                                        elif output_format == 'XRDML':
                                            file_content_to_download = generate_xrdml(applied_metadata_df,
                                                                                      df_to_convert,
                                                                                      filename=uploaded_file.name)
                                        elif output_format == 'RAW':
                                            file_content_to_download = generate_raw(applied_metadata_df, df_to_convert)

                                        if file_content_to_download:
                                            zf.writestr(new_filename, file_content_to_download)

                            st.download_button(
                                label=f"📦 Download All as {output_format} (.zip)",
                                data=zip_buffer.getvalue(),
                                file_name=f"converted_to_{output_format}.zip",
                                mime="application/zip",
                                type="primary",
                                use_container_width=True
                            )
                        else:
                            default_name = first_file.name.rsplit('.', 1)[0] + f'.{file_extension}'
                            download_filename = st.text_input("Enter filename for download:", default_name)

                            file_content_to_download = None
                            mime_type = 'application/octet-stream'
                            if output_format == 'RAS':
                                file_content_to_download = generate_ras(applied_metadata_df, data_df)
                                mime_type = 'text/plain'
                            elif output_format == 'XRDML':
                                file_content_to_download = generate_xrdml(applied_metadata_df, data_df,
                                                                          filename=first_file.name)
                                mime_type = 'application/xml'
                            elif output_format == 'RAW':
                                file_content_to_download = generate_raw(applied_metadata_df, data_df)

                            if file_content_to_download:
                                st.download_button(
                                    label=f"⬇️ Download as .{file_extension}",
                                    data=file_content_to_download,
                                    file_name=download_filename,
                                    mime=mime_type,
                                    type="primary",
                                    use_container_width=True
                                )

                with col2:
                    st.markdown("#### 📈 Diffraction Pattern")
                    fig = go.Figure(
                        go.Scatter(x=data_df['2Theta'], y=data_df['Intensity'], mode='lines', name='Intensity'))
                    fig.update_layout(title=f"Data from {first_file.name}", xaxis_title="2θ (°)",
                                      yaxis_title="Intensity", height=550, margin=dict(l=40, r=40, t=50, b=40))
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="XRD Data Converter")
    run_data_converter()
