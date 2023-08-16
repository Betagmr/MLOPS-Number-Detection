from functools import lru_cache
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from clearml import Model
from streamlit_drawable_canvas import st_canvas


def get_model() -> list[tuple[torch.nn.Module, str]]:
    my_models = Model.query_models(
        project_name="Number_detection/canary_training",
        tags=["production"],
    )

    model_list = []
    for model in my_models:
        model_path = model.get_local_copy()
        model_list.append((torch.jit.load(model_path), model.id))

    return model_list


def make_prediction(image: np.ndarray[Any, Any]) -> list[str]:
    model_list = get_model()
    tensor = torch.tensor(np.array([image])).float()

    result_list = []
    for model, id in model_list:
        result = model(tensor).tolist()
        value = np.argmax(result, axis=1)[0]
        result_list.append(f"{value} - {id}")

    return result_list


def main() -> None:
    drawing_mode = "freedraw"
    stroke_width = 15
    stroke_color = "#000000"
    bg_color = "#FFFFFF"
    realtime_update = True

    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=280,
        width=280,
        drawing_mode=drawing_mode,
        key="full_app",
    )

    if canvas_result.image_data is not None:
        original_image = canvas_result.image_data
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2GRAY)
        resized_image = 255 - cv2.resize(gray_image, (28, 28))

        image_array = np.array(resized_image.tolist())
        image_array = np.reshape(image_array, (1, 28, 28))
        st.image(resized_image)

        if st.button("Predict"):
            result = make_prediction(image_array)
            st.text(result)


if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Drawable Canvas Demo", page_icon=":rocket:")
    st.title("Drawable Canvas Demo")
    main()
