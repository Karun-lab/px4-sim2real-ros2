import cv2
from djitellopy import Tello


def get_orientation_data(drone):

    pitch = f"Pitch: {drone.get_pitch()}°"
    roll = f"Roll: {drone.get_roll()}°"
    yaw = f"Yaw: {drone.get_yaw()}°"

    vgx = f"Speed x: {drone.get_speed_x()} cm/s"
    vgy = f"Speed y: {drone.get_speed_y()} cm/s"
    vgz = f"Speed z: {drone.get_speed_z()} cm/s"

    agx = f"Acceleration x: {drone.get_acceleration_x()} cm/s²"
    agy = f"Acceleration y: {drone.get_acceleration_y()} cm/s²"
    agz = f"Acceleration z: {drone.get_acceleration_z()} cm/s²"

    print(
        pitch + '\n' +
        roll + '\n' +
        yaw + '\n' +
        vgx + '\n' +
        vgy + '\n' +
        vgz + '\n' +
        agx + '\n' +
        agy + '\n' +
        agz
    )


def get_status_data(drone):

    active_motor_time = drone.get_flight_time()
    battery = drone.get_battery()
    height = drone.get_height()
    tof = drone.get_distance_tof()

    templ = drone.get_lowest_temperature()
    temph = drone.get_highest_temperature()

    avg_temp = drone.get_temperature()
    barometer = drone.get_barometer()

    return (
        active_motor_time,
        battery,
        height,
        tof,
        templ,
        temph,
        avg_temp,
        barometer
    )


def process_tello_video(drone):

    frame_reader = drone.get_frame_read()

    frame_count = 0

    while True:

        frame = frame_reader.frame

        if frame is None:
            continue

        frame_count += 1

        if frame_count % 30 == 0:
            get_orientation_data(drone)

        (
            active_motor_time,
            battery,
            height,
            tof,
            templ,
            temph,
            avg_temp,
            barometer
        ) = get_status_data(drone)

        cv2.putText(
            frame,
            f"Flight Time: {active_motor_time} Seconds",
            (10, 20),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 255, 255),
            1,
            2
        )

        cv2.putText(
            frame,
            f"Battery: {battery}%",
            (10, 45),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 255, 255),
            1,
            2
        )

        cv2.putText(
            frame,
            f"Height: {height} cm",
            (10, 70),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 255, 255),
            1,
            2
        )

        cv2.putText(
            frame,
            f"TOF: {tof} cm",
            (10, 95),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (255, 255, 255),
            1,
            2
        )

        # cv2.putText(
        #     frame,
        #     f"Temperature: {avg_temp}°",
        #     (10, 120),
        #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #     1,
        #     (255, 255, 255),
        #     1,
        #     2
        # )

        # cv2.putText(
        #     frame,
        #     f"Barometer: {barometer} cm",
        #     (10, 145),
        #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #     1,
        #     (255, 255, 255),
        #     1,
        #     2
        # )

        #cv2.imshow("Frame", frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("RGB_CONVERTED", rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    drone.streamoff()
    cv2.destroyAllWindows()


def main():
    drone = Tello()
    drone.connect()
    print(f"Battery: {drone.get_battery()}%")
    drone.streamon()
    process_tello_video(drone)


if __name__ == '__main__':
    main()