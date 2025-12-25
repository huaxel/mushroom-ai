document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("mushroom-form");
    const resultBox = document.getElementById("result");
    const button = document.getElementById("predict-button");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        resultBox.style.display = "block";
        resultBox.style.background = "#fff3cd";
        resultBox.style.color = "#856404";
        resultBox.textContent = "Predicting...";

        button.disabled = true;
        button.innerHTML = '<span class="spinner"></span> Predicting...';

        try {
            console.time("Prediction fetch time"); // Start timer
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data),
            });

            console.timeEnd("Prediction fetch time"); // End timer and log elapsed time

            const result = await response.json();

            if (response.ok) {
                if (result.prediction === 1) {
                    resultBox.style.background = "#f8d7da"; // light red
                    resultBox.style.color = "#721c24"; // dark red
                    resultBox.textContent = "☠️ Poisonous!";
                } else if (result.prediction === 0) {
                    resultBox.style.background = "#d4edda"; // light green
                    resultBox.style.color = "#155724"; // dark green
                    resultBox.textContent = "✅ Edible";
                } else {
                    resultBox.style.background = "#ffeeba"; // fallback color
                    resultBox.style.color = "#856404";
                    resultBox.textContent = "Unknown prediction";
                }
            } else {
                resultBox.style.background = "#f8d7da";
                resultBox.style.color = "#721c24";
                resultBox.textContent =
                    "Error: " + (result.detail || "Unknown error");
            }
        } catch (err) {
            resultBox.style.background = "#f8d7da";
            resultBox.style.color = "#721c24";
            resultBox.textContent = "Error: " + err.message;
        } finally {
            button.disabled = false;
            button.textContent = "Predict";
        }
    });
});
