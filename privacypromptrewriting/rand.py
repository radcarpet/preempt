import random
import string

def _sanitize_date_Mmm_nnth(date_str):
    """Make a random date that looks like Apr 15th but with random strings,
    e.g. Dre 34th
    """
    # Extract month, day number, and suffix from the input date
    month, day = date_str.split()
    day_number = day[:-2]
    suffix = day[-2:]

    # Generate a random two-digit number for the new day
    new_day_number = random.randint(10, 99)

    # Generate a random three-letter month string
    new_month = random.choice(string.ascii_uppercase) + ''.join(random.choices(string.ascii_lowercase, k=2))

    # Combine the new month, new day number, and original suffix
    new_date_str = f"{new_month} {new_day_number}{suffix}"

    return new_date_str

def _sanitize_date_numeric(date_str):
    # gen random string of form dd/dd/dd
    new_date_str = ''.join(random.choices(string.digits, k=6))
    # insert / to make it look like dd/dd/dd
    new_date_str = new_date_str[:2] + '/' + new_date_str[2:4] + '/' + new_date_str[4:]
    return new_date_str

def sanitize_date(date_str):
    # with probab 0.5 return a random date of form dd/dd/dd
    # with probab 0.5 return a random date of form Mmm ddth
    return random.choice(
        [
            _sanitize_date_numeric(date_str),
            _sanitize_date_Mmm_nnth(date_str)
        ]
    )

def sanitize_zip(zip_code):
    # Pick random N from [4, 8]
    N = random.choice([4, 8])
    # make random numeric string of len N
    random_zip = ''.join(random.choices(string.digits, k=N))
    return random_zip

def sanitize_cc(card_number):
    # Remove spaces or hyphens if present
    clean_card_number = card_number.replace(" ", "").replace("-", "")

    # random integer from 8, 12, 20
    N = random.choice([8, 12, 20])
    # Check if the input is a valid credit card format (16 digits)
    if len(clean_card_number) == 16 and clean_card_number.isdigit():
        # Generate a random 16-digit number
        random_number = ''.join([str(random.randint(0, 9)) for _ in range(N)])

        # Format the random number in groups of 4 digits
        formatted_random_number = ' '.join([random_number[i:i+4] for i in range(0, N, 4)])

        return formatted_random_number
    else:
        return "Invalid credit card number"