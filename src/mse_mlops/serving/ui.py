from __future__ import annotations

import io
import os
import zipfile

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DOCTOR_PASSWORD = os.environ.get("DOCTOR_PASSWORD", "doctor123")

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")
st.title("Skin Cancer Detection")
st.caption("Upload a photo of a skin lesion and our AI will analyse whether it may be cancerous.")

if "doctor_auth" not in st.session_state:
    st.session_state.doctor_auth = False


def _doctor_login_form(key: str) -> None:
    st.warning("This section is for medical staff only.")
    password = st.text_input("Enter doctor password", type="password", key=f"pwd_{key}")
    if st.button("Unlock", key=f"unlock_{key}"):
        if password == DOCTOR_PASSWORD:
            st.session_state.doctor_auth = True
            st.rerun()
        else:
            st.error("Incorrect password.")


tab1, tab2, tab3 = st.tabs(["Skin Analysis", "Review & Label", "Bulk Dataset Upload"])

with tab1:
    st.header("Check Your Skin")
    st.write(
        "Upload a clear, close-up photo of the skin lesion you are concerned about. "
        "The AI will predict whether it appears benign or malignant."
    )
    st.info(
        "This tool is for informational purposes only and does not replace a professional medical diagnosis. "
        "Always consult a qualified dermatologist.",
    )

    uploaded = st.file_uploader(
        "Upload skin lesion image (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        key="predict_uploader",
    )

    if uploaded is not None:
        st.image(Image.open(uploaded), caption="Your uploaded image", use_container_width=True)

        if st.button("Analyse Image", type="primary"):
            uploaded.seek(0)
            with st.spinner("Analysing..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": (uploaded.name, uploaded.read(), "image/jpeg")},
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.stop()

            label = result["class"]
            confidence = result["confidence"]

            if label.lower() == "malignant":
                st.error(f"Result: Potentially Malignant - {confidence:.1%} confidence")
                st.write(
                    "The model flagged this lesion as potentially malignant. "
                    "Please consult a dermatologist as soon as possible."
                )
            else:
                st.success(f"Result: Likely Benign - {confidence:.1%} confidence")
                st.write(
                    "The model considers this lesion likely benign. "
                    "If you have any concerns, a professional check is always recommended."
                )

            st.subheader("Detailed probabilities")
            for class_name, probability in result["probabilities"].items():
                st.progress(probability, text=f"{class_name.capitalize()}: {probability:.1%}")

            st.caption(f"Reference ID: `{result['image_id']}`")

with tab2:
    st.header("Review & Label Predictions")

    if not st.session_state.doctor_auth:
        _doctor_login_form("tab2")
    else:
        st.caption("Review AI predictions and assign verified ground-truth labels.")

        col_refresh, col_logout = st.columns([5, 1])
        with col_refresh:
            if st.button("Refresh queue"):
                st.rerun()
        with col_logout:
            if st.button("Log out"):
                st.session_state.doctor_auth = False
                st.rerun()

        try:
            response = requests.get(f"{API_URL}/feedback", timeout=10)
            response.raise_for_status()
            entries = response.json()
        except Exception as exc:
            st.error(f"Could not fetch entries: {exc}")
            entries = []

        unlabeled = [entry for entry in entries if entry.get("label") is None and entry.get("prediction") is not None]

        if not unlabeled:
            st.info("No unlabeled predictions in the queue.")
        else:
            st.write(f"**{len(unlabeled)} unlabeled prediction(s) awaiting review**")
            for entry in unlabeled:
                confidence = entry.get("confidence") or 0.0
                with st.expander(
                    f"{entry.get('filename') or 'unknown'} - AI: {entry['prediction']} ({confidence:.1%})"
                ):
                    col_left, col_right = st.columns([2, 1])
                    with col_left:
                        st.write(f"**Reference ID:** `{entry['image_id']}`")
                        st.write(f"**AI prediction:** {entry['prediction']}")
                        st.write(f"**Confidence:** {confidence:.1%}")
                        st.write(f"**Submitted:** {entry['timestamp']}")
                    with col_right:
                        label_choice = st.selectbox(
                            "Verified label",
                            options=["benign", "malignant"],
                            key=f"label_{entry['image_id']}",
                        )
                        if st.button("Save label", key=f"submit_{entry['image_id']}"):
                            try:
                                response = requests.post(
                                    f"{API_URL}/feedback",
                                    json={
                                        "image_id": entry["image_id"],
                                        "label": label_choice,
                                        "source": "doctor_review",
                                    },
                                    timeout=10,
                                )
                                response.raise_for_status()
                                st.success(f"Saved as **{label_choice}**")
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed: {exc}")

with tab3:
    st.header("Bulk Dataset Upload")

    if not st.session_state.doctor_auth:
        _doctor_login_form("tab3")
    else:
        st.caption(
            "Upload a single labeled image, or a ZIP archive paired with a CSV/Excel label sheet "
            "to import many images at once."
        )

        if st.button("Log out", key="logout_tab3"):
            st.session_state.doctor_auth = False
            st.rerun()

        upload_mode = st.radio("Upload mode", ["Single image", "ZIP + label sheet"], horizontal=True)

        if upload_mode == "Single image":
            single_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="single_uploader")
            single_label = st.selectbox("Label", options=["benign", "malignant"], key="single_label")

            if single_file is not None:
                st.image(Image.open(single_file), caption="Preview", use_container_width=True)

                if st.button("Upload & Save", type="primary"):
                    single_file.seek(0)
                    with st.spinner("Uploading..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/upload-labeled",
                                files={"file": (single_file.name, single_file.read(), "image/jpeg")},
                                data={"label": single_label},
                                timeout=30,
                            )
                            response.raise_for_status()
                            result = response.json()
                            st.success(f"Saved! ID: `{result['image_id']}` - label: **{result['label']}**")
                        except Exception as exc:
                            st.error(f"Upload failed: {exc}")
        else:
            st.markdown(
                "**Expected format:**\n"
                "- A `.zip` file containing image files (JPG/PNG)\n"
                "- A CSV or Excel file with at minimum two columns: `filename` and `label`\n"
                "  - `filename` must match the image filename inside the ZIP (e.g. `img001.jpg`)\n"
                "  - `label` must be `benign` or `malignant`"
            )

            zip_file = st.file_uploader("ZIP archive", type=["zip"], key="zip_uploader")
            label_sheet = st.file_uploader(
                "Label sheet (CSV or Excel)",
                type=["csv", "xlsx", "xls"],
                key="sheet_uploader",
            )

            if zip_file is not None and label_sheet is not None:
                try:
                    df = pd.read_csv(label_sheet) if label_sheet.name.endswith(".csv") else pd.read_excel(label_sheet)
                except Exception as exc:
                    st.error(f"Could not read label sheet: {exc}")
                    st.stop()

                if "filename" not in df.columns or "label" not in df.columns:
                    st.error("Label sheet must have `filename` and `label` columns.")
                    st.stop()

                label_map = dict(zip(df["filename"].astype(str), df["label"].astype(str), strict=False))

                st.write(f"Label sheet loaded: **{len(label_map)} entries**")
                st.dataframe(df[["filename", "label"]].head(10), use_container_width=True)

                try:
                    zip_archive = zipfile.ZipFile(io.BytesIO(zip_file.read()))
                    image_names = [
                        name
                        for name in zip_archive.namelist()
                        if not name.endswith("/") and name.split("/")[-1].lower().endswith((".jpg", ".jpeg", ".png"))
                    ]
                except Exception as exc:
                    st.error(f"Could not open ZIP: {exc}")
                    st.stop()

                st.write(f"ZIP contains **{len(image_names)} image(s)**")

                matched = [
                    (name, label_map[name.split("/")[-1]])
                    for name in image_names
                    if name.split("/")[-1] in label_map
                ]
                unmatched = [name.split("/")[-1] for name in image_names if name.split("/")[-1] not in label_map]

                st.write(f"**{len(matched)} matched**, {len(unmatched)} unmatched")
                if unmatched:
                    with st.expander(f"Unmatched images ({len(unmatched)}) - will be skipped"):
                        st.write(unmatched)

                if matched and st.button(f"Upload {len(matched)} images", type="primary"):
                    progress = st.progress(0, text="Uploading...")
                    success_count = 0
                    errors = []

                    for index, (image_name, image_label) in enumerate(matched):
                        try:
                            image_bytes = zip_archive.read(image_name)
                            basename = image_name.split("/")[-1]
                            response = requests.post(
                                f"{API_URL}/upload-labeled",
                                files={"file": (basename, image_bytes, "image/jpeg")},
                                data={"label": image_label},
                                timeout=30,
                            )
                            response.raise_for_status()
                            success_count += 1
                        except Exception as exc:
                            errors.append(f"{image_name}: {exc}")

                        progress.progress(
                            (index + 1) / len(matched),
                            text=f"Uploading {index + 1}/{len(matched)}...",
                        )

                    if errors:
                        st.warning(f"Completed with {len(errors)} error(s):")
                        for error in errors:
                            st.text(error)
                    else:
                        st.success(f"Successfully uploaded **{success_count} images**!")
