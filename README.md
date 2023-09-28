# Challenge Script

This a python challenge to create a script that:

- Uses the NASA API to retrieve the Astronomy picture of the day
- Uses the Replicate API to retrieve a caption generated by a machine learning model
- Returns a dictionary with some info about the photo from nasa together with the caption generated by the machine learning model

## Current status

The app is single threaded thus far as Replicate's python client does not seem to support async/await. A good addition would be to add a spinner while the image recognition model generates a caption.

The captions can be funny depending on what image recognition model you use.

The app has some unit tests to specify the expected behavior if:

- the NASA API returns a code other than 200
- the NASA API returns a 200
- the Replicate API runs

It could use some more robust testing for API token failure, particularly for the replicate model


# Steps for running the app

After cloning the app, you need to:

1. Set up the environment variables:

   ```base
   cp sample.env .env
   ```

2. Create a virtual environment and activate it(recommended):

    ```base
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Install packages:

    ```base
    pip install -r requirments.txt
    ```

4. Run the app:

    ```base
    python challenge/script.py
    ```

5. Have some fun experimenting with [other image-to-text models](https://replicate.com/collections/image-to-text). Click on any of the links, select the API tab, copy the string that identifies the model from the `replicate.run(<model>, <input>)` snippet and use it to replace the AI_MODEL in your .env file.
