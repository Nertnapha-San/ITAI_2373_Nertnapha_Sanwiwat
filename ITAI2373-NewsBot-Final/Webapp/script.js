// NewsBot Intelligence System -- lightweight client-side behavior.
// No frameworks: a character counter and a submit-loading state.

document.addEventListener("DOMContentLoaded", function () {
    const textarea = document.getElementById("text");
    const charCount = document.getElementById("charCount");
    const analysisForm = document.getElementById("analysisForm");

    if (textarea && charCount) {
        const updateCount = () => {
            const count = textarea.value.length;
            charCount.textContent = count;
            charCount.style.color = count < 10 ? "#C0392B" : "#1E8A5F";
        };
        textarea.addEventListener("input", updateCount);
        updateCount();
    }

    if (analysisForm) {
        analysisForm.addEventListener("submit", function (e) {
            const text = (textarea ? textarea.value : "").trim();
            if (text.length < 10) {
                e.preventDefault();
                alert("Please enter at least 10 characters to analyze.");
                return;
            }
            const button = analysisForm.querySelector("button[type='submit']");
            if (button) {
                button.classList.add("loading");
                button.disabled = true;
            }
        });
    }

    const askForm = document.getElementById("askForm");
    if (askForm) {
        askForm.addEventListener("submit", function () {
            const button = askForm.querySelector("button[type='submit']");
            if (button) {
                button.classList.add("loading");
                button.disabled = true;
            }
        });
    }
});
