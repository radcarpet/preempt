# to create q-a dataset involving date, zip, cc

You are an expert at creating question-answer datasets, and also expert in creating fictitious, non-existent credit card numbers.
Your job is to create a dataset consisting of 10 examples,
from an e-commerce domain, where each example contains:

P: passage context of 2-3 sentences, involving a FULL FAKE credit card number, a date, and a zip code
Q: question about the passage context
A: The correct answer to the question

The dataset should be exactly in the above format, and you should separate
the examples by 2 newlines.

Do not number the examples, and write all the examples in a markdown block.
Remember:
*  I want to see FULL FAKE credit card numbers in the passages, even if you have to make them up.
* All dates must be in the form: 8/15/2021 etc
* Dates should NOT refer to credit card expiration dates.
* All dates should have FORMAT like Jan 15th, Feb 10th etc (NO YEAR, and 3-letter month)
* Questions CANNOT contain SENSITIVE info like Date, Zip, or Credit Card
* ANSWERS CAN contain SENSITIVE info like Date, Zip, or Credit Card, .e.g if Question is "Which credit card was used", then Answer would contain Credit card.

