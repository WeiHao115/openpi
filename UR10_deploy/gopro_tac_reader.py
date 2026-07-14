
from cv_bridge import CvBridge
import rospy
import numpy as np
import sys
import time
import threading
import cv2
import threadpoolctl
import os
from PIL import Image
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import PoseStamped, WrenchStamped
import os
import threading
import numpy as np
from collections import deque
import rospy
from geometry_msgs.msg import WrenchStamped
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("/home/k202/gsmini_ws/src")
try:
    import gelSight_SDK.examples.gsdevice as gsdevice
except ImportError:
    raise ImportError("无法导入 gelSight_SDK,请检查系统路径 /home/ywl/gsmini_ws/src 是否存在。")

def process_and_resize_frame(frame_bgr, target_size):
    """
    对内存中的 NumPy 图像阵列进行中心裁剪和缩放
    """
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    short_edge = min(h, w)
    start_x = (w - short_edge) // 2
    start_y = (h - short_edge) // 2

    img_cropped = frame_bgr[start_y:start_y+short_edge, start_x:start_x+short_edge]
    img_resized = cv2.resize(img_cropped, target_size)

    return img_resized

class GelSightManager:
    def __init__(self,
                 dev1_id="GelSight Mini R0B 2DAT-2LMZ",
                 dev2_id="GelSight Mini R0B 2DPF-C3HB"):

        self.dev1 = gsdevice.Camera(dev1_id)
        self.dev2 = gsdevice.Camera(dev2_id)
        self.dev1.connect()
        self.dev2.connect()

        self.frame_1 = None
        self.frame_2 = None
        self.timestamp_1 = 0.0
        self.timestamp_2 = 0.0

        self.running = True

        # 为两路视频分配独立的锁
        self.lock_1 = threading.Lock()
        self.lock_2 = threading.Lock()

        # 为两路视频分配独立的读取线程，防止单一设备超时阻塞另一设备
        self.thread_1 = threading.Thread(target=self._update_loop_1, daemon=True)
        self.thread_2 = threading.Thread(target=self._update_loop_2, daemon=True)

        self.thread_1.start()
        self.thread_2.start()

        print("GelSightManager 初始化完成，双路独立读取线程已启动。")

    def _update_loop_1(self):
        while self.running:
            f1 = self.dev1.get_raw_image()
            if f1 is not None:
                with self.lock_1:
                    self.frame_1 = f1
                    self.timestamp_1 = time.time()

    def _update_loop_2(self):
        while self.running:
            f2 = self.dev2.get_raw_image()
            if f2 is not None:
                with self.lock_2:
                    self.frame_2 = f2
                    self.timestamp_2 = time.time()

    def get_tactile_frame(self):
        out_f1, out_f2 = None, None

        with self.lock_1:
            if self.frame_1 is not None:
                out_f1 = self.frame_1.copy()

        with self.lock_2:
            if self.frame_2 is not None:
                out_f2 = self.frame_2.copy()

        # 返回时间戳以最新的一路为准，或自行调整逻辑
        return out_f1, out_f2, max(self.timestamp_1, self.timestamp_2)

    def release(self):
        self.running = False
        self.thread_1.join()
        self.thread_2.join()


class GoproManager:
    def __init__(self, device_id=6, width=1280, height=720, fps=15):
        cv2.setNumThreads(1)
        threadpoolctl.threadpool_limits(1)

        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频设备: {device_id}")

        self.running = True
        self.lock = threading.Lock()

        self._buffer_frame = None
        self.current_frame = None
        self.kernel_timestamp = 0.0

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            grabbed = self.cap.grab()
            if not grabbed:
                continue

            v4l2_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = v4l2_msec / 1000.0
            ret, self._buffer_frame = self.cap.retrieve()

            if ret and self._buffer_frame is not None:
                # Modified by DK
                # ---------------- 新增：安全遮罩逻辑 ----------------
                h, w, _ = self._buffer_frame.shape

                # 设定预期的遮罩高度和宽度
                # 使用 min() 防止设定的像素值超过实际图像尺寸引发越界崩溃
                mask_h = min(60, h)
                mask_w = min(80, w)
                mask_color = (0, 0, 0)

                # 绘制左上角遮罩
                self._buffer_frame[0:mask_h, 0:mask_w] = mask_color
                # 绘制右上角遮罩
                self._buffer_frame[0:mask_h, w-mask_w:w] = mask_color
                # ----------------------------------------------------

                with self.lock:
                    self.current_frame = self._buffer_frame.copy()
                    self.kernel_timestamp = timestamp_sec


    def get_latest_frame(self):
        with self.lock:
            if self.current_frame is not None:
                # 传入 NumPy 数组，返回处理后的 NumPy 数组
                processed_frame = process_and_resize_frame(self.current_frame, (224, 224))
                return processed_frame, self.kernel_timestamp
        return None, 0.0


    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


class RealsenseRosManager:
    def __init__(self, topic_name="/camera/color/image_raw", save_dir=""):
        try:
            rospy.init_node("data_record_node", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass

        self.save_dir = save_dir
        self.bridge = CvBridge()

        self.lock = threading.Lock()
        self.current_frame = None
        self.timestamp = 0.0

        # 订阅话题
        self.sub_rs = rospy.Subscriber(topic_name, RosImage, self._callback, queue_size=10)
        print(f"ROS RealSense 订阅节点初始化完成，监听话题: {topic_name}")

    def _callback(self, msg):
        try:
            # 绕过 cv_bridge，直接解析数据
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

            # ROS 默认通常是 RGB，OpenCV 需要 BGR
            if msg.encoding == "rgb8":
                cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                cv_img = img_np # 如果已经是 bgr8 则直接赋值

            t = msg.header.stamp.to_sec()

            with self.lock:
                self.current_frame = cv_img.copy()
                self.timestamp = t
        except Exception as e:
            print(f"RealSense 图像转换解析失败: {e}")

    def get_latest_frame(self):
        with self.lock:
            if self.current_frame is not None:
                processed_frame = process_and_resize_frame(self.current_frame, (640, 480))
                return processed_frame, self.timestamp

                # return self.current_frame.copy(), self.timestamp
        return None, 0.0

    def release(self):
        if hasattr(self, 'sub_rs'):
            self.sub_rs.unregister()


class ForceTorqueManager:
    def __init__(self, topic_name="/ur10_force_sensor/wrench", save_dir="./", median_window=5, mean_window=3):
        self.save_dir = save_dir
        self.img_save_path = os.path.join(save_dir, "force_torque_comparison.png")
        self.lock = threading.Lock()

        self.force_hz = 200
        self.timestamp = 0.0
        self.current_wrench = None      # 当前时刻滤波前的六维力
        self.filtered_wrench = None     # 当前时刻滤波后的六维力

        self.median_window = median_window
        self.mean_window = mean_window
        self.raw_buffer = deque(maxlen=median_window)   # 当前时刻滤波前六维力列表,
        self.median_buffer = deque(maxlen=mean_window)  # 当前时刻滤波后六维力列表

        # 实时内存缓存：用于最后退出时绘制对比图
        self.history_timestamps = []
        self.history_original = []      # 保存整个轨迹中所有原始六维力数据
        self.history_filtered = []      # 保存整个轨迹中所有滤波后六维力数据
        self.sub_ft = rospy.Subscriber(topic_name, WrenchStamped, self._callback_realtime, queue_size=10)


    def _callback_realtime(self, msg: WrenchStamped):
        wrench_array = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ], dtype=np.float32)
        self.filtered_wrench = wrench_array.tolist()


    def _callback_old(self, msg: WrenchStamped):
        wrench_array = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ], dtype=np.float32)

        # self.raw_buffer保存原始数据，队列
        # self.median_buffer保存滤波后的数据，队列
        with self.lock:
            self.timestamp = msg.header.stamp.to_sec()
            self.current_wrench = wrench_array.tolist()     # 当前时刻滤波前的六维力
            self.raw_buffer.append(wrench_array)
            current_median = np.median(self.raw_buffer, axis=0)
            self.median_buffer.append(current_median)
            current_mean = np.mean(self.median_buffer, axis=0)
            self.filtered_wrench = current_mean.tolist()    # 当前时刻滤波后的六维力
            rospy.sleep(1 / self.force_hz)


    def _callback(self, msg: WrenchStamped):
        wrench_array = np.array([
            msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
            msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
        ], dtype=np.float32)

        with self.lock:
            self.timestamp = msg.header.stamp.to_sec()
            self.current_wrench = wrench_array.tolist()     # 当前时刻滤波前的六维力
            self.raw_buffer.append(self.current_wrench)
            raw_buffer = np.array(self.raw_buffer)

            if raw_buffer.shape[0] != self.median_window:
                extend_num = self.median_window - raw_buffer.shape[0]
                concat_tensor = np.broadcast_to(raw_buffer[0][None], (extend_num, 6))
                raw_buffer = np.concatenate((concat_tensor, raw_buffer), axis=0)

            raw_buffer_filter = self.apply_edge_preserving_filter(raw_buffer, 31, 9)
            self.filtered_wrench = raw_buffer_filter[-1][None]  # [6] -> [1 6]
            rospy.sleep(1 / self.force_hz)


    def apply_edge_preserving_filter(self, force_data, median_window=31, mean_window=9):
        if force_data is None or len(force_data) == 0:
            return force_data

        # 将 numpy 数组转换为 DataFrame 以利用高效的 rolling 算子
        df = pd.DataFrame(force_data)
        df_med = df.rolling(window=median_window, center=True, min_periods=1).median()
        df_smooth = df_med.rolling(window=mean_window, center=True, min_periods=1).mean()
        return df_smooth.values.astype(np.float32)


    def get_ori_filtered_wrench(self):
        with self.lock:
            # 必须先判断是否为 None，如果是，直接返回 None，禁止执行 np.array()
            if self.filtered_wrench is None:
                return None
            # 只有在确信有数据的情况下，才进行 numpy 数组转换
            return np.array(self.filtered_wrench, dtype=np.float32)


    def get_filtered_wrench(self):
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            force_state = self.get_ori_filtered_wrench()
            # 3. 只有当数据不为 None 时才打印并跳出循环
            if force_state is not None:
                return force_state
            rate.sleep()


    def record_current_data(self, sys_timestamp=None):
        """
        实时将当前帧数据追加到内存的 List 中，避免实机运行时读写磁盘或绘图导致阻塞。
        """
        with self.lock:
            if self.filtered_wrench is not None and self.current_wrench is not None:
                # 记录时间戳可以使用 ROS 时间戳 self.timestamp，也可以使用传入的系统时间戳
                t = sys_timestamp if sys_timestamp is not None else self.timestamp
                self.history_timestamps.append(t)
                self.history_original.append(self.current_wrench)
                self.history_filtered.append(self.filtered_wrench)



    def plot_and_save_force_comparison(self):
        """
        在程序退出时被调用，一次性绘制滤波前后对比图。
        """
        with self.lock:
            if not self.history_timestamps:
                print("未录制到任何力觉数据，取消绘图。")
                return

            timestamps = np.array(self.history_timestamps)
            original_data = np.array(self.history_original)
            filtered_data = np.array(self.history_filtered)

        print(f"正在生成六维力滤波对比图，总计数据量: {len(timestamps)} 帧...")

        ft_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
        y_units = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']

        # 计算执行进度百分比 (%)，防止除以 0 的安全处理
        time_range = timestamps[-1] - timestamps[0]
        if time_range > 0:
            progress = ((timestamps - timestamps[0]) / time_range) * 100
            x_label = 'Execution Progress (%)'
        else:
            progress = timestamps - timestamps[0]
            x_label = 'Relative Time (s)'

        fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

        for i, ax in enumerate(axes):
            axis_name = ft_cols[i]
            ax.plot(progress, original_data[:, i], label=f'Original {axis_name}', color='blue', alpha=0.5, linewidth=1.2)
            ax.plot(progress, filtered_data[:, i], label=f'Filtered {axis_name}', color='red', alpha=0.8, linewidth=1.5)
            ax.set_ylabel(f'{axis_name} ({y_units[i]})')
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.6)

        axes[-1].set_xlabel(x_label)
        plt.tight_layout()
        plt.savefig(self.img_save_path, dpi=300)
        plt.close()
        print(f"对比图已成功保存至: {self.img_save_path}")


    def release(self):
        if hasattr(self, 'sub_ft'):
            self.sub_ft.unregister()
        self.plot_and_save_force_comparison()



def main():
    try:
        rospy.init_node("gopro_tac_reader_node", anonymous=True)
    except rospy.exceptions.ROSException:
        pass

    save_dir = "multimodal_records"
    save_dir = "multimodal_records"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 实例化触觉与视觉管理器
    # manager = GelSightManager()
    # 启用 GoproManager 以避免主线程被 read() 阻塞
    try:
        gopro_manager = GoproManager(device_id=6, width=224, height=224, fps=30)
        # force_manager = ForceTorqueManager(topic_name="/landian_wrench", median_window=31, mean_window=9)
    except RuntimeError as e:
        print(f"警告: {e}")
        gopro_manager = None

    # try:
    #     force_manager = ForceTorqueManager(topic_name="/landian_wrench", median_window=31, mean_window=9)
    # except RuntimeError as e:
    #     print(f"警告: {e}")
    #     force_manager = None

    # print("等待力矩传感器连接...")

    # while force_manager.get_filtered_wrench() is None:
    #     time.sleep(0.1)

    last_save_time = time.time()
    save_interval = 10.0
    save_counter = 0

    # 此时数据已经就绪，可以成功记录下启动瞬间的第一帧数据
    # force_manager.record_current_data(last_save_time)

    print('11111')

    print("开始获取传感器与相机图像。在图像窗口按 'q' 键退出程序。")

    try:
        while True:
            # # 1. 核心修复：只要传感器启用了，每一轮循环都高频记录当前帧的力觉数据
            # if force_manager is not None:
            #     force_manager.record_current_data()  # 实时追加到内存 List

            gopro_frame, gp_timestamp = None, 0.0
            if gopro_manager is not None:
                gopro_frame, gp_timestamp = gopro_manager.get_latest_frame()

            if gopro_frame is not None:
                cv2.imshow("GoPro Camera", gopro_frame)

                current_time = time.time()

                # 2. 10 秒定时控制块：这里只负责保存 GoPro 图像，不要把力觉数据的记录卡在这里
                if current_time - last_save_time >= save_interval:
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))

                    save_gopro = gopro_frame.copy()
                    cv2.putText(save_gopro, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # filename_gopro = os.path.join(save_dir, f"gopro_{save_counter:04d}.png")
                    # cv2.imwrite(filename_gopro, save_gopro)

                    # print(f"[{time.strftime('%H:%M:%S')}] 已保存图像快照，序号: {save_counter:04d}")

                    last_save_time = current_time
                    save_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n检测到键盘中断信号。")
    finally:
        # manager.release()
        if gopro_manager is not None:
            gopro_manager.release()
        # if force_manager is not None:
        #     force_manager.release()
        cv2.destroyAllWindows()
        print("程序已退出。")


if __name__ == "__main__":
    main()


    # rospy.init_node("force_manager_test_node", anonymous=True)
    # force_manager = ForceTorqueManager(topic_name="/wrench", median_window=5, mean_window=5)
    # for i in range(2000):
    #     force_value = force_manager.get_filtered_wrench()
    #     force_manager.record_current_data()
    #     print(force_value)
    # force_manager.release()

    # all_numpy = []
    # force_manager = ForceTorqueManager(topic_name="/landian_wrench", save_dir="", median_window=310)
    # while not rospy.is_shutdown():
    #     filtered_wrench = force_manager.get_filtered_wrench()
    #     all_numpy.append(filtered_wrench)
    #     print(np.array(all_numpy).shape)
    #     np.savetxt("/home/k202/lerobot/UR10e_deploy/multimodal_records/force_manager.txt", np.array(all_numpy))


    # def plot_and_save_force_comparison(timestamps, original_data, filtered_data, save_path):
    #     """
    #     绘制并保存滤波前后六维力的对比图像。
    #     """
    #     ft_cols = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    #     y_units = ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']

    #     # 计算执行进度百分比 (%)
    #     progress = ((timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])) * 100

    #     fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    #     for i, ax in enumerate(axes):
    #         axis_name = ft_cols[i]
    #         ax.plot(progress, original_data[:, i], label=f'Original {axis_name}', color='blue', alpha=0.5, linewidth=1.2)
    #         ax.plot(progress, filtered_data[:, i], label=f'Filtered {axis_name}', color='red', alpha=0.8, linewidth=1.5)
    #         ax.set_ylabel(f'{axis_name} ({y_units[i]})')
    #         ax.legend(loc='upper right')
    #         ax.grid(True, linestyle='--', alpha=0.6)
    #         # if 'F' in axis_name:
    #         #     ax.set_ylim([-5.0, 5.0])  # 力统一显示 ±5N 范围
    #         # else:
    #         #     ax.set_ylim([-1.0, 1.0])  # 力矩统一显示 ±1Nm 范围
    #     axes[-1].set_xlabel('Execution Progress (%)')
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=300)
    #     plt.close()

    # all_force = np.loadtxt("/home/k202/lerobot/UR10e_deploy/multimodal_records/force_test.txt")
    # timestamps = np.array(range(0, all_force.shape[0]))
    # plot_and_save_force_comparison(timestamps, all_force, all_force, "/home/k202/lerobot/UR10e_deploy/multimodal_records/forcenew.jpg")



    # timestamps = np.array(range(0, 310))
    # raw_buffer = deque(maxlen=310)
    # for i in range(100):
    #     raw_buffer.append(np.random.rand(6))
    # raw_buffer = np.array(raw_buffer)
    # if raw_buffer.shape[0] != 310:
    #     extend_num = 310 - raw_buffer.shape[0]
    #     concat_tensor = np.broadcast_to(raw_buffer[0][None], (extend_num, 6))
    #     raw_buffer = np.concatenate((concat_tensor, raw_buffer), axis=0)

    # raw_buffer_filter = apply_edge_preserving_filter(raw_buffer, 31, 9)
    # print(raw_buffer)
    # print(raw_buffer_filter)
    # plot_and_save_force_comparison(timestamps, raw_buffer, raw_buffer_filter, "/home/k202/lerobot/UR10e_deploy/force.jpg")
