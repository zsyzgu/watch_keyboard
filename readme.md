# 可运行的程序

(1) baseline.py

[Usage] python baseline.py save_folder

baseline程序，基于鼠标点击的文本输入法（带纠错）

(2) entry.py

[Usage] python baseline.py save_folder

主程序，基于双手环的任意物体表面文本输入法（带纠错）

(3) register.py

校准程序，每个用户在使用主程序（entry.py）之前，都可以使用这个校准程序，使得主程序适用于该用户的手的大小。

# 需要解释的类

(1) FingerTracker类

追踪一只手的五个手指，如果要追踪两只手，则要实例化两个对象。每帧会更新的数据有：

1. self.fintertips 五指指尖在图像上的坐标
2. self.is_touch 五指指尖是否触碰到了桌面
3. self.is_touch_down/is_touch_down 是否是点击/抬起事件
4. self.endpoints 五指在桌子上的坐标（厘米）
5. self.corr_endpoints 五指在桌子上的坐标（根据校准个性化、归一化）
6. self.palm_line 手掌下边缘的线的行数

函数：

1. def run(self, image): 每帧将图像数据传入，上面的五指追踪数据就会更新。
2. def output(self, title=''): 返回illustration图像，title会打在图像上。

(2) Decoder类

1. def predict(self, data, inputted): 根据用户落点位置（data），预测用户所需单词并返回。
 