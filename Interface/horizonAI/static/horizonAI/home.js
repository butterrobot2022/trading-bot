document.addEventListener('DOMContentLoaded', function () {
    
    const form = document.querySelector('.bottom-waitlist-form');

    form.addEventListener('click', function(event) {
        // Prevent the default link behavior
        event.preventDefault();
        // Scroll to the top of the page
        window.scrollTo(0, 0);
    });
    const waitlistForm = document.querySelector('.waitlist-email');
    const emailInput = document.querySelector('.waitlist-email-input');

    waitlistForm.addEventListener('submit', function (event) {
        event.preventDefault();

        const email = emailInput.value;

        // Get the CSRF token value from the HTML form
        const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]').value;
        console.log(csrfToken);
        // Make an AJAX request to the Django view
        fetch('/join_waitlist', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,  // Include CSRF token from the HTML form
            },
            body: JSON.stringify({ email }),
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message); // Display the response message (you can handle it differently)
            if (data.message === 'Success') {
                // Optionally, do something when the signup is successful (e.g., show a success message)
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing your request.');
        });
    });
    
});
