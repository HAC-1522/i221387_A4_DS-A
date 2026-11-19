import argparse, time, numpy as np, onnxruntime as ort
from PIL import Image
from torchvision import transforms

MEAN=(0.5071,0.4867,0.4408); STD=(0.2675,0.2565,0.2761)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True); p.add_argument("--image",required=True)
    a=p.parse_args()
    sess=ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    tf=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),
                           transforms.Normalize(MEAN,STD)])
    x=tf(Image.open(a.image).convert("RGB")).unsqueeze(0).numpy().astype(np.float32)
    name=sess.get_inputs()[0].name
    for _ in range(5): sess.run(None,{name:x})
    start=time.perf_counter()
    logits=sess.run(None,{name:x})[0]
    elapsed=(time.perf_counter()-start)*1000
    probs=np.exp(logits[0]-np.max(logits[0])); probs=probs/probs.sum()
    top=np.argsort(probs)[-5:][::-1]
    print("CPU ONNX Runtime inference")
    print(f"Latency: {elapsed:.3f} ms")
    for i in top: print(f"class_index={i:02d} confidence={probs[i]:.4f}")
if __name__=="__main__": main()
