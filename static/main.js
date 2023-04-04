let image_path = "";

document.getElementById('upload-image').addEventListener('submit', async function(event) {
    event.preventDefault();

    const imageInput = document.getElementById('image');

    const formData = new FormData();
    formData.append('image', imageInput.files[0]);

    const response = await fetch('/upload_image', { method: 'POST', body: formData });

    const data = await response.json();

    const imageUploadResultDiv = document.getElementById('image-upload-result');
    if (data.success) {
        const timestamp = new Date().getTime();
        imageUploadResultDiv.innerHTML = `<img src="/static/images/${data.image_path}?${timestamp}" alt="Uploaded image" width="150" height="150">`;
        image_path = data.image_path;
    } else {
        imageUploadResultDiv.innerHTML = `<p>Image upload failed</p>`;
    }
});

document.getElementById('generate-text').addEventListener('submit', function (event) {
    event.preventDefault(); // prevent the form from submitting in the default way
  
    // get the values from the form inputs
    const input = document.getElementById('input').value;
    const maxTokens = document.getElementById('maxTokens').value;
    const temperature = document.getElementById('temperature').value;
    const topK = document.getElementById('topK').value;
    const topP = document.getElementById('topP').value;
    const repeatPenalty = document.getElementById('repeatPenalty').value;
    const seed = document.getElementById('seed').value;
 
    // make the POST request to the server to generate text and get an SSE connection ID
    fetch('/generate_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_path: image_path,
            input: input,
            max_tokens: maxTokens,
            temperature: temperature,
            top_k: topK,
            top_p: topP,
            repeat_penalty: repeatPenalty,
            seed: seed
        })
    })
    .then(response => response.json())
    .then(data => {
        const sseConnectionId = data.sseConnectionId;
        // create an EventSource with the returned SSE connection ID
        const eventSource = new EventSource(`/generate_text_sse?connection_id=${sseConnectionId}`);
        console.log(eventSource)
        let output = '';

        eventSource.onmessage = function (event) {
            let full_str = event.data;
            // trim the whitespace from the start and end of the string
            full_str = full_str.trim();
            console.log(full_str);
            if (full_str === '[DONE]') {
                eventSource.close();
                // display the generated text on the page
                const testGeneratedResultDiv = document.getElementById('text-generated-result');
                testGeneratedResultDiv.textContent = full_str;
            } else {
                const testGeneratedResultDiv = document.getElementById('text-generated-result');
                testGeneratedResultDiv.textContent = full_str;
            }
        };

        eventSource.onerror = function (error) {
            // console log the details of the error
            console.log("eventSource.onerror:", error);
        };
    })
    .catch(error => {
        console.log('Request failed:', error);
    });
});
