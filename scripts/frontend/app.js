let total = 0;
let healthy = 0;
let diseased = 0;

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");

imageInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        previewImage.src = URL.createObjectURL(file);
    }
});

async function predict() {

    let file = imageInput.files[0];
    if (!file) {
        alert("Select image first");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    let response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    let data = await response.json();

    document.getElementById("resultCard").classList.remove("hidden");

    document.getElementById("diseaseName").innerText =
        data.disease.replaceAll("_", " ");

    document.getElementById("confidenceText").innerText =
        "Confidence: " + (data.confidence * 100).toFixed(2) + "%";

    document.getElementById("progressBar").style.width =
        (data.confidence * 100) + "%";

    total++;

    if (data.disease.includes("healthy"))
        healthy++;
    else
        diseased++;

    document.getElementById("totalScans").innerText = total;
    document.getElementById("healthyCount").innerText = healthy;
    document.getElementById("diseasedCount").innerText = diseased;
}
