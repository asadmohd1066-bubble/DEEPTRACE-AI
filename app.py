import streamlit as st
from PIL import Image, ImageChops, ImageEnhance
from detector import analyze_image
import io

st.set_page_config(
    page_title="DeepTrace AI",
    page_icon="🔍",
    layout="wide"
)
st.title("🔍 DeepTrace AI")
st.subheader("AI-Powered Digital Image Forensics")
st.caption("DETECT • ANALYSE • VERIFY")
st.write(
    "Analyse images using AI and forensic techniques "
    "to identify possible signs of manipulation."
)

st.divider()

uploaded_file = st.file_uploader(
    "📁 Upload an image for analysis",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("""
    <div style="
        background:linear-gradient(135deg,#FFFFFF,#E7F2ED);
        border:1px solid #B9D8CC;
        border-radius:16px;
        padding:16px;
        margin:15px 0;
    ">
        <h3 style="color:#126B68;margin:0;">
            🖼️ Uploaded Image
        </h3>
        <p style="color:#52716B;margin:5px 0 0;">
            Image successfully loaded for forensic inspection
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:
        st.markdown("""
        <div style="
            background:#FFFFFF;
            border:1px solid #B9D8CC;
            border-radius:14px;
            padding:18px;
        ">
            <h3 style="color:#126B68;">📊 Image Information</h3>
        """, unsafe_allow_html=True)

        st.write("**Width:**", image.width, "pixels")
        st.write("**Height:**", image.height, "pixels")
        st.write(
            "**File size:**",
            round(uploaded_file.size / 1024, 2), "KB"
        )
        st.write("**Format:**", uploaded_file.type)

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.header("2. Metadata Analysis")

    exif_data = Image.open(uploaded_file).getexif()

    if exif_data:
        st.success("EXIF metadata found")
        for key, value in exif_data.items():
            st.write(f"Tag {key}: {value}")
    else:
        st.warning("No EXIF metadata found")

    st.caption(
        "Missing metadata does not prove an image is fake."
    )

    st.divider()

    st.header("3. Error Level Analysis (ELA)")

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)

    compressed = Image.open(buffer).convert("RGB")

    difference = ImageChops.difference(image, compressed)
    difference = ImageEnhance.Contrast(difference).enhance(10)

    st.image(
        difference,
        caption="ELA Visualization",
        use_container_width=True
    )

    st.caption(
        "ELA highlights compression differences. "
        "These differences are not proof of manipulation."
    )

    st.divider()

    st.header("4. AI-Powered Image Analysis")

    if st.button("🚀 Analyse Image with AI"):

        with st.spinner("AI model is analysing..."):

            try:
                fake_probability = analyze_image(image)

                fake_probability = max(
                    0.0, min(1.0, float(fake_probability))
                )

                real_probability = 1.0 - fake_probability

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "AI-generated score",
                        f"{fake_probability * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Real-image score",
                        f"{real_probability * 100:.2f}%"
                    )

                st.progress(fake_probability)

                st.warning(
                    "These are model-estimated scores, "
                    "not proof that an image is real or fake. "
                    "Model performance depends on image type."
                )

                st.success("AI analysis completed!")

            except Exception as error:
                st.error(f"Analysis error: {error}")

else:
    st.info("👆 Upload an image to begin forensic analysis.")

st.divider()

st.caption(
    "DeepTrace AI | First-Year Innovation Hackathon 2026"
)
# ==========================================
# 5. DEEPTRACE AI - ADVANCED TESTING LAB
# ==========================================

st.divider()

st.header("5. Advanced Testing Lab")
st.write(
    "Upload multiple labelled images to evaluate "
    "the model and compare 5-crop analysis."
)

uploaded_test_images = st.file_uploader(
    "Upload Real and AI test images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    key="testing_lab_upload"
)

if uploaded_test_images:

    st.subheader("Assign Ground Truth Labels")

    test_labels = {}

    for index, uploaded in enumerate(uploaded_test_images):

        col1, col2 = st.columns([2, 1])

        with col1:
            st.write(f"**{uploaded.name}**")

        with col2:
            test_labels[index] = st.selectbox(
                "Actual image type",
                ["Real", "AI-generated"],
                key=f"test_label_{index}"
            )

    if st.button("Run Complete Testing", type="primary"):

        try:
            from detector_experiment import analyze_multicrop

            results = []

            progress = st.progress(0)
            status = st.empty()

            for index, uploaded in enumerate(uploaded_test_images):

                status.write(
                    f"Testing {index + 1}/{len(uploaded_test_images)}: "
                    f"{uploaded.name}"
                )

                image = Image.open(uploaded).convert("RGB")

                original_ai = float(analyze_image(image))

                crop_result = analyze_multicrop(image)

                mean_ai = crop_result["mean_ai"]

                actual = test_labels[index]
                predicted = (
                    "AI-generated" if mean_ai >= 0.50 else "Real"
                )

                results.append({
                    "Image": uploaded.name,
                    "Actual": actual,
                    "Predicted": predicted,
                    "Original AI %": round(original_ai * 100, 2),
                    "5-Crop Mean AI %": round(mean_ai * 100, 2),
                    "Center AI %": round(
                        crop_result["center_ai"] * 100, 2
                    ),
                    "Maximum Crop AI %": round(
                        crop_result["max_ai"] * 100, 2
                    ),
                    "Crop Scores": ", ".join(
                        f"{score * 100:.2f}%"
                        for score in crop_result["crop_scores"]
                    )
                })

                progress.progress(
                    (index + 1) / len(uploaded_test_images)
                )

            status.success("Testing completed successfully!")

            # Performance calculations
            tp = sum(
                r["Actual"] == "AI-generated"
                and r["Predicted"] == "AI-generated"
                for r in results
            )

            tn = sum(
                r["Actual"] == "Real"
                and r["Predicted"] == "Real"
                for r in results
            )

            fp = sum(
                r["Actual"] == "Real"
                and r["Predicted"] == "AI-generated"
                for r in results
            )

            fn = sum(
                r["Actual"] == "AI-generated"
                and r["Predicted"] == "Real"
                for r in results
            )

            total = len(results)

            accuracy = (tp + tn) / total if total else 0
            precision = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) else 0
            )

            st.divider()
            st.subheader("Testing Performance")

            m1, m2, m3, m4 = st.columns(4)

            m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
            m2.metric("Precision", f"{precision * 100:.2f}%")
            m3.metric("Recall", f"{recall * 100:.2f}%")
            m4.metric("F1 Score", f"{f1 * 100:.2f}%")

            st.subheader("Confusion Matrix")

            st.table([
                {
                    "Actual Class": "AI-generated",
                    "Predicted AI": tp,
                    "Predicted Real": fn
                },
                {
                    "Actual Class": "Real",
                    "Predicted AI": fp,
                    "Predicted Real": tn
                }
            ])

            st.subheader("Individual Image Results")
            st.dataframe(results, use_container_width=True)

            st.success(
                f"Correct predictions: {tp + tn}/{total}"
            )

            st.info(
                "Prediction threshold: 50% 5-crop mean AI score. "
                "Metrics describe only the uploaded labelled test set; "
                "they do not establish general real-world accuracy."
            )

        except Exception as error:
            st.error(f"Testing Lab Error: {error}")

else:
    st.info("Upload multiple images to start advanced testing.")