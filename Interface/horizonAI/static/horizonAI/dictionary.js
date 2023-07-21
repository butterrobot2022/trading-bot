document.addEventListener("DOMContentLoaded", function () {
    const searchInput = document.querySelector(".form-control");
    const showTermDiv = document.querySelector(".show-term");
    const wordbaseDiv = document.querySelector(".wordbase");
    const wordbaseList = document.querySelectorAll(".wordbase li");

    searchInput.addEventListener("input", function () {
        const searchTerm = searchInput.value.trim().toLowerCase();
        let matchingTerms = [];

        if (searchTerm !== "") {
            wordbaseList.forEach(function (item) {
                const wordname = item.querySelector(".wordname").innerText.toLowerCase();
                if (wordname.includes(searchTerm)) {
                    matchingTerms.push(item.innerHTML);
                }
            });
        }

        showTermDiv.innerHTML = matchingTerms.length
            ? `<ul>${matchingTerms.join("")}</ul>`
            : "No matching terms found.";

        showTermDiv.style.display = matchingTerms.length ? "block" : "none";
        wordbaseDiv.style.display = matchingTerms.length ? "none" : "block";
    });
});