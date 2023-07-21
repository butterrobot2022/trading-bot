function searchBooks() {
    const searchQuery = document.getElementById("search-input").value.toLowerCase();
    const allBookCategories = document.querySelectorAll(".books .list-books");
    const searchedBooksDiv = document.querySelector(".searched-books");
    const mainBookContent = document.querySelector(".main-book-content");
    const allBooks = document.querySelector('.books');

    if (searchQuery.trim() === "") {
        searchedBooksDiv.style.display = "none";
        mainBookContent.style.display = "flex";
        // allBooks.style.display = 'block';
        allBookCategories.forEach(bookCategory => {
            bookCategory.style.display = "block";
        });
        return;
    }

    let searchedBooksHTML = '';
    let hasMatch = false;

    allBookCategories.forEach(bookCategory => {
        const books = bookCategory.querySelectorAll("li a.title-book");

        let categoryHasMatch = false;
        let categoryHTML = '';

        books.forEach(book => {
            const bookName = book.textContent.toLowerCase();

            if (bookName.includes(searchQuery)) {
                hasMatch = true;
                categoryHasMatch = true;
                categoryHTML += `<div class='searched-books'><a href="#" class="title-book">${book.innerHTML}</a></div><br>`;
            }
        });

        if (categoryHasMatch) {
            bookCategory.style.display = "block";
        } else {
            bookCategory.style.display = "none";
        }

        searchedBooksHTML += categoryHTML;
    });

    if (hasMatch) {
        searchedBooksDiv.innerHTML = searchedBooksHTML;
        searchedBooksDiv.style.display = "block";
        // searchedBooksDiv.className = 'main-book-content'
        mainBookContent.style.display = "none";
    } else {
        searchedBooksDiv.style.display = "none";
        mainBookContent.style.display = "block";
    }
}
