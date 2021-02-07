import socketio
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sio = socketio.Client()
sio.connect('http://127.0.0.1:4567')

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

@sio.on('visualization')
def visualization(data):
    global steering_angle, throttle, speed

    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = float(data["speed"])

        print("steering_angle: {} \t throttle: {} \t speed: {}".\
            format(steering_angle, throttle, speed))
    else:
        steering_angle, throttle, speed = 0,0,0

# This function is called periodically from FuncAnimation
def animate(i, xs, ys):
    # Add x and y to lists
    xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
    ys.append(speed)

    # Limit x and y lists to n items
    xs = xs[-80:]
    ys = ys[-80:]

    # Draw x and y lists
    ax.clear()
    ax.plot(xs, ys)
    ax.tick_params(axis='x', which='major', labelsize=2)

    # Format plot
    # bottom, top = -1.0, 23.0
    # y_ticks = np.arange(bottom, top, 2)
    # plt.yticks(y_ticks)
    # plt.ylim(bottom, top)
    plt.xticks(rotation=90, ha='right')
    plt.subplots_adjust(bottom=0.30)
    plt.title('Speed over Time')
    plt.ylabel('Speed (MPH)')


if __name__ == '__main__':
    steering_angle, throttle, speed = 0.0, 0.0, 0.0

    # Create figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    xs = []
    ys = []

    # Set up plot to call animate() function periodically
    ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys), interval=100)
    plt.show()

