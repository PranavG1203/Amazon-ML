1. Setting Up Tools and Libraries
The script uses following tools:
-Pandas: Helps read and write CSV files.
-pytesseract: This is the tool that reads text from images (called OCR, or Optical Character Recognition).
-Python Imaging Library(PIL): Manages images, like opening them or enhancing their quality.
-requests: Fetches images from the web by their URLs.
-Regular Expressions (re): Finds and formats the information extracted from the images.
-OpenCV (cv2): A toolkit to process images, such as converting them to black-and-white, resizing, or sharpening them.
-fuzzywuzzy: Used for "fuzzy matching," which checks how closely the extracted information from the image matches the expected value.


2. Cleaning and Formatting Text
Cleaning the text: Removes unnecessary characters or symbols that could interfere with the comparison.
Normalizing numbers: Ensures that numbers are in a standard format, for example turning "1,000" into "1000" or adjusting decimals.

3. Fuzzy Matching
The script uses a fuzzy matching technique to compare the extracted information (like "500 grams") with the correct value. It checks how similar the two are. If the match is good enough (above 80% similarity), the script accepts it.

4. How the Script Extracts Information from Images
-Fetch the Image: It starts by downloading the image from the URL provided in the CSV file.
-Image Processing: The script prepares the image for text extraction by:
	Converting it to black-and-white to make the text stand out.
	Enlarging the image for better clarity.
	Enhancing the contrast to make the text sharper.
	Removing noise and blurring to make the letters more readable.
-Extract Text: After all these steps, it uses pytesseract to extract the text from the image.

5. Formatting the Extracted Text
The script looks for specific patterns in the text depending on the type of information (like weight or voltage). For instance, it might search for "500 grams" or "230 volts" and extract the number and unit. It then converts the unit into a standardized form (like converting "g" to "gram").

6. For training process: 
Comparing Extracted Information with the Expected Value
The script reads the CSV file that contains a list of images, the type of feature to extract (like weight or size), and the correct value for comparison.
For each image:
It extracts the text.
Cleans and formats it.
Extracts the relevant feature (like weight).
Compares it with the expected value using fuzzy matching.
It keeps track of how many correct matches it found and calculates the accuracy of its predictions.

7. All the above processing in executed in "Processor.py".

8. This file is then imported in "main.py" which consists of the 'test.csv' dataset loaded and passed to the module as arguments for prediction
