from pymavlink import mavutil
import socket
import sys
import serial
from serial.tools import list_ports

def list_serial_ports():

    """
    Reads all serial port connections and returns device and description. 
    If there are no connections returns (None, None) instead of 
    (device, description)

    input: None
    return: (device, description)
    """
    serial_ports = list_ports.comports()

    if len(serial_ports) == 0:
        print("No serial ports found")
        return None, None

    print("Available serial ports: ")
    for port_info in serial_ports:
        print(f"{port_info.device} - {port_info.description}")

    return port_info.device, port_info.description


device, id = list_serial_ports()

print(f"attempting to connect to device {id}")
drone_cnt = mavutil.mavlink_connection(device, baud=57600)

print(drone_cnt)

drone_cnt.wait_heartbeat()

print(f"targeting system {drone_cnt.target_system}, targeting component {drone_cnt.target_component}")
drone_cnt.mav.request_data_stream_send(
    drone_cnt.target_system,
    drone_cnt.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
    1, #hz stream rate
    1  #bool (0 or 1) start stream
)


for i in range(10):
    #msg_name = 'SMART_BATTERY_INFO' #ONBOARD_COMPUTER_STATUS ( #390 ) # OPEN_DRONE_ID_LOCATION HL_FAILURE_FLAG # ESTIMATOR_STATUS (mag_ratio)
    #WIND_COV HOME_POSITION VIBRATION ESC_STATUS(voltage current rpm time_usec)
    msg_name = 'BATTERY_STATUS'
    msg=drone_cnt.recv_match(type=msg_name, blocking=True)
    if i < 1:
        print("mavlink message attributes")
        #print([member for member in msg.__dir__() if 'battery' in member or 'current' in member])
        print(msg.__dir__())
        print("---------------")
        print("mavlink message object discription")
        print('lel')

    if msg != None:
        print(f"RECIEVED {msg.type}")
        print(f"VOLTS: {msg.voltages}, CURRENT CONSUMED: {msg.current_consumed}, BATTERY_CAPACITY {msg.battery_remaining}")
        #print(f"VOLTS: {msg.voltages}, CURRENT: {msg.current}, CAPACITY {msg.remaining}")
        print(f"TYPE: {msg.type}")
        print(f"MODE: {msg.mode}")
        print(f"TIME LEFT: {msg.time_remaining}")
        print(f"BATTERY FUNCTION {msg.battery_function}")
    else:
        print("NO BATTERY DATA")

    msg_ekf = drone_cnt.recv_match(type='ESTIMATOR_STATUS', blocking=True)

    if msg_ekf != None:
        #print(f"RECIEVED {msg_ekf.type}")
        print(f"MAG ERROR: {msg_ekf.mag_ratio}")
    else:
        print("NO EKF DATA")

    msg_esc = drone_cnt.recv_match(type='ESC_STATUS', blocking=True)

    if msg_esc != None:
        #print(f"RECIEVED {msg_esc.type}")
        print(f"ESC CURRENT: {msg_esc.current}")
    else:
        print("NO ESC DATA")
