# 测试文本情感识别是否可用
# 直接运行这个文件即可！

# 导入你的主代码
from text_emotion_recognizer import get_text_emotion_api

# ===================== 测试句子 =====================
print("="*50)
print("开始测试文本情感识别...")
print("="*50)

# 测试1：正面情绪
test1 = "I am so happy today! I got a great score."
res1 = get_text_emotion_api(test1)
print("测试1 - 正面文本:", test1)
print("识别结果:", res1)

print("-"*30)

# 测试2：负面情绪
test2 = "I feel very sad and disappointed."
res2 = get_text_emotion_api(test2)
print("测试2 - 负面文本:", test2)
print("识别结果:", res2)

print("-"*30)

# 测试3：愤怒
test3 = "I am so angry about this thing!"
res3 = get_text_emotion_api(test3)
print("测试3 - 愤怒文本:", test3)
print("识别结果:", res3)

print("-"*30)

# 测试4：中性
test4 = "The weather is normal today."
res4 = get_text_emotion_api(test4)
print("测试4 - 中性文本:", test4)
print("识别结果:", res4)

print("="*50)
print("✅ 测试完成！代码可以正常使用！")
print("="*50)