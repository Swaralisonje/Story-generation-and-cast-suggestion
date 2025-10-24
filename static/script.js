// static/script.js

document.addEventListener('DOMContentLoaded', () => {
    const suggestBtn = document.getElementById('suggest-cast-btn');
    const resultsContainer = document.getElementById('cast-suggestion-results');
    const storyTextContainer = document.getElementById('full-story-text');
    const loadingSpinner = document.getElementById('loading-spinner');

    if (suggestBtn) {
        suggestBtn.addEventListener('click', async () => {
            const story = storyTextContainer.textContent.trim();

            if (!story) {
                alert("Could not find the story text.");
                return;
            }

            // Show loading spinner and disable button
            loadingSpinner.classList.remove('d-none');
            suggestBtn.disabled = true;
            suggestBtn.textContent = 'Analyzing...';
            resultsContainer.innerHTML = ''; // Clear previous results

            try {
                const response = await fetch('/suggest_cast', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ story: story })
                });

                if (!response.ok) {
                    throw new Error(`Server responded with status: ${response.status}`);
                }

                const castData = await response.json();
                displayCast(castData);

            } catch (error) {
                console.error("Error fetching cast suggestions:", error);
                resultsContainer.innerHTML = `<div class="alert alert-danger">An error occurred while suggesting the cast. Please try again.</div>`;
            } finally {
                // Hide loading spinner and re-enable button
                loadingSpinner.classList.add('d-none');
                suggestBtn.disabled = false;
                suggestBtn.textContent = 'Suggest Cast for this Story 🎭';
            }
        });
    }

    function displayCast(characters) {
        if (!characters || characters.length === 0) {
            resultsContainer.innerHTML = `<div class="alert alert-info">Could not identify any characters in the story.</div>`;
            return;
        }

        let html = '<h2 class="mb-4 text-center">Cast Suggestions</h2>';

        characters.forEach(char => {
            html += `
            <div class="card cast-card">
                <div class="card-header cast-card-header">
                    Character: ${char.character_name}
                </div>
                <div class="card-body">
                    <p><strong>Predicted Gender:</strong> ${char.predicted_gender || 'Unknown'}</p>
                    <p><strong>Detected Age:</strong> ${char.age || 'Not mentioned'}</p>
                    <p><strong>Story Genre Context:</strong> ${char.genre_from_story || 'Unknown'}</p>

                    <h6 class="mt-4">Top Actor Suggestions:</h6>
            `;

            if (char.suggestions && char.suggestions.length > 0) {
                html += `
                    <table class="table table-bordered table-striped mt-2">
                        <thead>
                            <tr>
                                <th>Actor Name</th>
                                <th>Famous Genre</th>
                                <th>Rating</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                char.suggestions.forEach(s => {
                    html += `
                        <tr>
                            <td>${s.actor_name || 'N/A'}</td>
                            <td>${s.actor_genre || 'N/A'}</td>
                            <td>${s.actor_rating !== null ? s.actor_rating : 'N/A'}</td>
                        </tr>
                    `;
                });
                html += `
                        </tbody>
                    </table>
                `;
            } else {
                html += `<p class="text-muted">No suitable actor suggestions found.</p>`;
            }
            html += `</div></div>`; // Close card-body and card
        });
        
        // Use insertAdjacentHTML to add the new content after the spinner
        loadingSpinner.insertAdjacentHTML('afterend', html);
    }
});