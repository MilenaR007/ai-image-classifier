import gradio as gr
from transformers import pipeline
import torch

print("Ladowanie modelu...")
classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=-1,
    dtype=torch.float32
)
print("Model gotowy!")

def rozpoznaj_obraz(image):
    if image is None:
        return {}, "Nie przeslano obrazu."

    try:
        wyniki = classifier(image, top_k=5)
        etykiety = {wynik["label"]: float(wynik["score"]) for wynik in wyniki}

        najlepszy = wyniki[0]
        pewnosc = round(najlepszy["score"] * 100, 1)
        opis = f"Na obrazie prawdopodobnie znajduje sie: **{najlepszy['label']}** ({pewnosc}% pewnosci)."

        return etykiety, opis

    except Exception as e:
        return {}, f"Blad podczas analizy: {str(e)}"

interfejs = gr.Interface(
    fn=rozpoznaj_obraz,
    inputs=gr.Image(type="pil", label="Wgraj zdjecie"),
    outputs=[
        gr.Label(num_top_classes=3, label="Rozpoznane kategorie"),
        gr.Markdown(label="Opis")
    ],
    title="Sztuczna Inteligencja Rozpoznaje Obrazy",
    description="Wgraj dowolne zdjecie (np. psa, kota, samochodu), a model AI zgadnie, co na nim jest!",
    flagging_mode="never"
)

interfejs.queue(max_size=3).launch()