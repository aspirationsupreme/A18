import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 加载训练好的模型
model = tf.keras.models.load_model('best_model_resnet50.h5')

# 类别标签
class_names = ['夹杂物', '补丁', '划痕', '其他缺陷']


# 预处理上传的图片
def preprocess_image(img):
    img = img.resize((1600, 256))  # 调整图片大小以符合模型输入要求
    img_array = np.array(img) / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0)  # 扩展维度，符合模型输入要求
    return img_array


# 预测函数
def predict(image):
    # 预处理图片
    img_array = preprocess_image(image)

    # 进行预测
    predictions = model.predict(img_array)

    # 输出每个类别的预测概率
    result = {class_names[i]: float(predictions[0][i]) for i in range(len(class_names))}

    # 设定阈值0.5来确定预测类别
    predicted_classes = [class_names[i] for i, pred in enumerate(predictions[0]) if pred >= 0.5]

    return result, predicted_classes


# 设置 Streamlit 页面标题
st.title('缺陷分类预测')

# 上传图片的文件上传控件
uploaded_image = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # 打开并显示上传的图片
    image = Image.open(uploaded_image)
    st.image(image, caption='上传的图片', use_column_width=True)

    # 预测按钮
    if st.button('进行预测'):
        with st.spinner('正在预测...'):
            # 调用预测函数
            probabilities, predicted_classes = predict(image)

            # 显示预测结果
            st.subheader("每个类别的预测概率:")
            for class_name, prob in probabilities.items():
                st.write(f"{class_name}: {prob:.2f}")

            # 显示预测类别
            if predicted_classes:
                st.subheader("预测类别:")
                st.write(", ".join(predicted_classes))
            else:
                st.write("未能预测出缺陷类型。")

# 运行 Streamlit 应用
