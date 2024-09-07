const express = require('express');
const axios = require('axios');
const cors = require('cors');  // Import cors middleware
const app = express();
const port = 3000;

app.use(cors());  // Enable CORS for all routes
app.use(express.json());

app.get('/check-status', async (req, res) => {
    const { url } = req.query;

    try {
        const response = await axios.get(url, {
            maxRedirects: 0,
            validateStatus: function (status) {
                return status >= 200 && status < 600;
            }
        });

        let statusMessage = '';

        if (response.status === 200) {
            statusMessage = `Good URL: ${response.status} OK`;
        } else if (response.status >= 300 && response.status < 400) {
            statusMessage = `Redirect: ${response.status}`;
        } else if (response.status >= 400 && response.status < 500) {
            statusMessage = `Client Error: ${response.status}`;
        } else if (response.status >= 500) {
            statusMessage = `Server Error: ${response.status}`;
        } else {
            statusMessage = `Unexpected Status: ${response.status}`;
        }

        res.json({ statusMessage });

    } catch (error) {
        console.error('Error fetching the URL:', error);
        res.json({ statusMessage: 'Error fetching the URL' });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}/`);
});
