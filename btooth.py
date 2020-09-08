
import serial

port = "/dev/ttyACM0"
baud = 9600

ser = serial.Serial(port, baud, timeout=1)
if ser.isOpen():
    print(ser.name + ' is open...')

while True:
	ch = raw_input('input:  ')    
	if ch == 'R':
		ser.write('R')
	elif ch == 'L':
		ser.write('L')
	elif ch == 'S':
		ser.write('S')
	else :
		print 'oops'
