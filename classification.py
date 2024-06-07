import streamlit as st
import pickle
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from skimage.feature import local_binary_pattern, hog


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# CIFAR-100 的 100 个类别名称
class_names = [
    "苹果", "水族馆鱼", "婴儿", "熊", "海狸", "床", "蜜蜂", "甲虫", "自行车", "瓶子", "碗",
    "男孩", "桥", "公共汽车", "蝴蝶", "骆驼", "罐头", "城堡", "毛虫", "牛", "椅子", "黑猩猩",
    "时钟", "云", "蟑螂", "沙发", "蟹", "鳄鱼", "杯", "恐龙", "海豚", "大象", "扁鱼", "森林",
    "狐狸", "女孩", "哈姆斯特", "房屋", "袋鼠", "键盘", "灯", "草坪割草机", "豹", "狮子", "蜥蜴",
    "龙虾", "男人", "枫树", "摩托车", "山", "鼠", "蘑菇", "橡树", "橙子", "兰花", "河狸", "棕榈树",
    "梨", "皮卡车", "松树", "平原", "盘子", "罂粟花", "豪猪", "负鼠", "兔子", "浣熊", "鲇鱼", "道路",
    "火箭", "玫瑰", "海", "海狮", "鲨鱼", "鼹鼠", "臭鼬", "摩天大楼", "蜗牛", "蛇", "蜘蛛", "松鼠",
    "有轨电车", "向日葵", "甜椒", "桌子", "坦克", "电话", "电视", "老虎", "拖拉机", "火车", "鳟鱼",
    "郁金香", "乌龟", "衣柜", "鲸鱼", "柳树", "狼", "女人", "虫"
]

st.title("Streamlit可视化AI应用")
st.sidebar.title("设置")
st.sidebar.header("作者信息")
st.sidebar.write("作者：王银平")
st.sidebar.write("学号：2109120114")
st.sidebar.header("算法选择")
algorithm_select = st.sidebar.selectbox("选择算法",
                                        ["灰度化", "二值化", "边缘检测", "神经网络模型分类", "LBP朴素贝叶斯模型分类",
                                         "LBP KNN模型分类", "LBP逻辑回归模型分类", "HOG朴素贝叶斯模型分类",
                                         "HOG KNN模型分类", "HOG逻辑回归模型分类", "SIFT朴素贝叶斯模型分类",
                                         "SIFT KNN模型分类", "SIFT逻辑回归模型分类"])
st.sidebar.header("日期和时间")
date_time = st.sidebar.date_input("当前日期")
time = st.sidebar.time_input("当前时间")
st.sidebar.title("音乐播放")
music_select = st.sidebar.selectbox("选择音乐", ["《稻香》", "《搁浅》", "《以父之名》"])
if music_select == "《稻香》":
    audio_file = open('daoxiang.mp3', 'rb')
elif music_select == "《搁浅》":
    audio_file = open('Step Aside.mp3', 'rb')
elif music_select == "《以父之名》":
    audio_file = open('In The Name Of Father.mp3', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg', autoplay=True)
st.write("欢迎来到我们的网站！您可以选择背景音乐")
uploaded_file = st.file_uploader("选择图像文件", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    algorithm = algorithm_select
    if algorithm == "灰度化":
        image_gray = image.convert("L")
        st.image(image_gray, caption="灰度化后的图像", use_column_width=True)
    elif algorithm == "二值化":
        image_binary = image.convert("1")
        st.image(image_binary, caption="二值化后的图像", use_column_width=True)
    elif algorithm == "边缘检测":
        image_edge = np.array(image)
        image_edge = np.where(image_edge > 128, 255, 0)
        st.image(image_edge, caption="边缘检测后的图像", use_column_width=True)
    else:
        st.image(image, caption="上传的图像", use_column_width=True)
        if algorithm == "神经网络模型分类":
            # 加载并使用神经网络模型
            net = Net()
            net.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
            net.eval()
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            output = net(image_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted[0].item()
            if 0 <= predicted_class < len(class_names):
                st.write(f"分类结果：{class_names[predicted_class]}")
            else:
                st.write("无法识别的类别")
        else:
            # 加载并使用传统机器学习模型
            model_path = {
                "LBP朴素贝叶斯模型分类": 'lbp_nb_model.pkl',
                "LBP KNN模型分类": 'lbp_knn_model.pkl',
                "LBP逻辑回归模型分类": 'lbp_lr_model.pkl',
                "HOG朴素贝叶斯模型分类": 'hog_nb_model.pkl',
                "HOG KNN模型分类": 'hog_knn_model.pkl',
                "HOG逻辑回归模型分类": 'hog_lr_model.pkl',
                "SIFT朴素贝叶斯模型分类": 'sift_nb_model.pkl',
                "SIFT KNN模型分类": 'sift_knn_model.pkl',
                "SIFT逻辑回归模型分类": 'sift_lr_model.pkl'
            }
            model_file = model_path[algorithm]
            with open(model_file, 'rb') as f:
                model = pickle.load(f)


            # 执行适当的特征提取
            def extract_lbp_features(image):
                gray_image = image.convert('L')
                image_np = np.array(gray_image)
                lbp = local_binary_pattern(image_np, 8 * 3, 3, method='uniform')
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 * 3 + 3),
                                         range=(0, 8 * 3 + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-6)
                # LBP features may not always give the same number of features, make sure to resize
                hist.resize(50, refcheck=False)  # Ensure 50 features
                return hist


            def extract_hog_features(image):
                gray_image = image.convert('L')
                image_np = np.array(gray_image)
                features, _ = hog(image_np, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, multichannel=False)
                return features


            def extract_sift_features(image):
                # You need to implement actual SIFT feature extraction method here
                gray_image = np.array(image.convert('L'))
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(gray_image, None)
                if descriptors is None:
                    descriptors = np.zeros((1, 128))
                descriptors.resize(50, refcheck=False)  # Ensure 50 features
                return descriptors.flatten()


            transform_to_input = {
                "LBP朴素贝叶斯模型分类": extract_lbp_features,
                "LBP KNN模型分类": extract_lbp_features,
                "LBP逻辑回归模型分类": extract_lbp_features,
                "HOG朴素贝叶斯模型分类": extract_hog_features,
                "HOG KNN模型分类": extract_hog_features,
                "HOG逻辑回归模型分类": extract_hog_features,
                "SIFT朴素贝叶斯模型分类": extract_sift_features,
                "SIFT KNN模型分类": extract_sift_features,
                "SIFT逻辑回归模型分类": extract_sift_features,
            }

            features = transform_to_input[algorithm](image)
            input_data = features.reshape(1, -1)

            predicted_class_probs = model.predict_proba(input_data)
            predicted_idx = np.argmax(predicted_class_probs, axis=1)[0]
            if 0 <= predicted_idx < len(class_names):
                st.write(f"分类结果：{class_names[predicted_idx]}")
            else:
                st.write("无法识别的类别")