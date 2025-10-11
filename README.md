<h2><b>Project</b></h2>
Developed a face recognition system using deep learning, providing multiple interfaces to showcase its functionality:<br><br>
<b>Backend (Jupyter Notebook):</b> Processes datasets, generates embeddings, and enables real-time recognition via webcam.<br>
<b>Desktop GUI (PyQt5):</b> Interactive application for uploading images and getting instant recognition results.<br>
<b>Web App (Streamlit):</b> Simple online interface for image upload and face matching.<br><br>

The system can be used for security, attendance management, identity verification, and personal authentication.n.


## Features

- **Upload Images**: Upload images for face detection and recognition.
- **Interactive UI**: User-friendly web interface using Streamlit.
- **Face Matching**: Matches uploaded faces against a pre-trained dataset.



<h2><b>Technologies & Frameworks</b></h2>

**Programming Language**: Python 3.x

**Deep Learning Framework**: PyTorch

**Computer Vision**: OpenCV, PIL

**Face Detection & Recognitionv**: MTCNN, InceptionResnetV1 (FaceNet)

**GUI Framework**: PyQt5

**Web Framework**: Streamlit

**Data Handling**: torchvision.datasets, DataLoader


<h2><b>Project Structure</b></h2>

```bash
FaceNet--based-Face-Recognition-app/Project/
│
├── app.ipynb
├── app.py
├── streamlit.py
├── data.pt
├── Model.pt
├── Readme.md
└── photos/
```
<h2><b>Installation and Setup</b></h2>

<h2><b>1. Clone the Repository</b></h2>

<pre> 
git clone https://github.com/Mubashirakhan03/HealthCare-Vision-Assistant.git cd HealthCare-Vision-Assistant 
</pre>



<h2><b>Installation & Setup</b></h2>


<h2><b>1. Clone the Repository</b></h2>
<pre>
git clone <repository-url>
cd FaceNet--based-Face-Recognition-app
</pre>



<h2><b>2. Create a Virtual Environment</b></h2>
<pre>
# For Windows
python -m venv venv
venv\Scripts\activate
</pre>


<h2><b>3. Install Dependencies</b></h2>
<pre>
pip install torch torchvision facenet-pytorch opencv-python pillow PyQt5 streamlit
</pre>


## 4. Prepare the Dataset</b></h2>
- Place individual images in the `photos/` folder.
- Folder names should correspond to person names (e.g., `photos/Alice`, `photos/Bob`).

## 5. Generate Embeddings
Run `app.ipynb` to process the images and save the embeddings to `data.pt`.


## Contact
For any queries: [mubashirakhan1001@gmail.com](mailto:mubashirakhan1001@gmail.com)
