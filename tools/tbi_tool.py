from pathlib import Path
import re
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
import pytz
import boto3
from botocore.config import Config
import os
from docx import Document
from io import BytesIO

class S3Client:
    def __init__(self):
        self.s3 = boto3.client('s3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION'),
            endpoint_url=os.getenv('AWS_ENDPOINT_URL'),
            config=Config(s3={"addressing_style": "virtual"})
        )
        self.bucket_name = os.getenv('BUCKET_NAME')

    def get_document_text_for_numerology(self, indicator_type: str, number: int, tci_number: int = None, bci_number: int = None) -> str:
        """
        Download file from S3 based on indicator type and value.

        Args:
            indicator_type (str): Indicator type, e.g. 'life_path', 'personal_day', 'personal_year', etc.
            number (int): Number value.

        Returns:
            str: Text content of the document.
        """
        # Map number types to folder and file naming conventions
        # Based on actual bucket structure: numerology_trader/{folder}/{file}
        base_folder = "trader_behavior_index"
        
        if indicator_type == "ami":
            folder = f"{base_folder}/ami"
            file_name = f"ami_{number}.docx"
        elif indicator_type == "dai":
            folder = f"{base_folder}/dai"
            file_name = f"dai_{number}.docx"
        elif indicator_type == "mri":
            folder = f"{base_folder}/mri"
            file_name = f"mri_{number}.docx"
        elif indicator_type == "ppa":
            folder = f"{base_folder}/ppa"
            file_name = f"ppa_{number}.docx"
        elif indicator_type == "spi":
            folder = f"{base_folder}/spi"
            file_name = f"spi_{number}.docx"
        elif indicator_type == "edi":
            folder = f"{base_folder}/edi"
            file_name = f"edi_{number}.docx"
        elif indicator_type == "mpi":
            folder = f"{base_folder}/mpi"
            file_name = f"mpi_{number}.docx"
        elif indicator_type == "cmi":
            folder = f"{base_folder}/cmi"
            file_name = f"cmi_{number}.docx"
        elif indicator_type == "ri":
            folder = f"{base_folder}/ri"
            file_name = f"ri_{number}.docx"
        elif indicator_type == "sai":
            folder = f"{base_folder}/sai"
            file_name = f"sai_{number}.docx"
        elif indicator_type == "wmi":
            folder = f"{base_folder}/wmi"
            file_name = f"wmi_{number}.docx"
        elif indicator_type == "ari":
            folder = f"{base_folder}/ari"
            file_name = f"ari_{number}.docx"
        elif indicator_type == "tci":
            # TCI documents are indexed by TCI order (1..4)
            folder = f"{base_folder}/tci_{tci_number}"
            file_name = f"tci_{number}.docx"
        elif indicator_type == "bci":
            # BCI documents are indexed by BCI order (1..4)
            folder = f"{base_folder}/bci_{bci_number}"
            file_name = f"bci_{number}.docx"
        else:
            # For unsupported types, return a placeholder
            return f"Document for {indicator_type} with value {number} is not available."

        file_key = f"{folder}/{file_name}"
        print(f"[S3 Numerology] Fetching: bucket={self.bucket_name}, key={file_key}")

        if self.s3 is None:
            raise ValueError("S3 client is not initialized.")

        try:
            # print(f"🔍 Attempting to fetch: {file_key} from bucket: {self.bucket_name}")
            
            # Check if bucket exists and is accessible
            try:
                self.s3.head_bucket(Bucket=self.bucket_name)
                # print(f"✅ Bucket {self.bucket_name} is accessible")
            except Exception as bucket_error:
                print(f"Bucket access issue: {bucket_error}")
                return f"Cannot access bucket {self.bucket_name}: {bucket_error}"
            
            # Try to get the object
            try:
                obj = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
                # print(f"Object retrieved successfully")
                
                # Check if Body exists and is readable
                if "Body" not in obj:
                    print(f"No Body in S3 response: {obj.keys()}")
                    return f"Response does not have Body: {list(obj.keys())}"
                
                body = obj["Body"]
                if body is None:
                    print(f"Body is None")
                    return f"Body of the response is None"
                
                # Read the content
                try:
                    file_content = body.read()
                    # print(f"File content read: {len(file_content)} bytes")
                except Exception as read_error:
                    print(f"Error reading body: {read_error}")
                    return f"Error reading body: {read_error}"
                
                # Check if content is valid
                if not file_content:
                    print(f"File content is empty")
                    return f"File content is empty"
                
                # Try to parse as document
                try:
                    doc = Document(BytesIO(file_content))
                    # print(f"Document parsed successfully")
                    
                    # Extract text
                    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
                    if not paragraphs:
                        print(f"No text content in document")
                        return f"Document does not have text content"
                    
                    text = "\n".join(paragraphs)
                    # print(f"Text extracted: {len(text)} characters")
                    return text
                    
                except Exception as doc_error:
                    print(f"Error parsing document: {doc_error}")
                    return f"Error parsing document: {doc_error}"
                    
            except Exception as obj_error:
                print(f"Error getting object: {obj_error}")
                return f"Error getting object: {obj_error}"
                
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return f"Unexpected error: {str(e)}"

class TBICalculator:
    """
    TBI Calculator - Tính toán các chỉ số hành vi giao dịch
    
    Dựa trên tên, ngày sinh và thông tin cá nhân để tính toán
    các chỉ số TBI (Trading Behavior Intelligence)
    """

    # Mapping alphabet to numbers (similar to numerology system)
    ALPHABET = {
        # Basic alphabet
        'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
        'J': 1, 'K': 2, 'L': 3, 'M': 4, 'N': 5, 'O': 6, 'P': 7, 'Q': 8, 'R': 9,
        'S': 1, 'T': 2, 'U': 3, 'V': 4, 'W': 5, 'X': 6, 'Y': 7, 'Z': 8,

        # Vowels with diacritics
        'Ă': 1, 'Â': 1, 'Ê': 5, 'Ô': 6, 'Ơ': 6,

        # Consonants with diacritics
        'Đ': 4,

        # Vowels with tone
        'Á': 1, 'À': 1, 'Ả': 1, 'Ã': 1, 'Ạ': 1,
        'Ắ': 1, 'Ằ': 1, 'Ẳ': 1, 'Ẵ': 1, 'Ặ': 1,
        'Ấ': 1, 'Ầ': 1, 'Ẩ': 1, 'Ẫ': 1, 'Ậ': 1,
        'É': 5, 'È': 5, 'Ẻ': 5, 'Ẽ': 5, 'Ẹ': 5,
        'Ế': 5, 'Ề': 5, 'Ể': 5, 'Ễ': 5, 'Ệ': 5,
        'Í': 9, 'Ì': 9, 'Ỉ': 9, 'Ĩ': 9, 'Ị': 9,
        'Ó': 6, 'Ò': 6, 'Ỏ': 6, 'Õ': 6, 'Ọ': 6,
        'Ố': 6, 'Ồ': 6, 'Ổ': 6, 'Ỗ': 6, 'Ộ': 6,
        'Ớ': 6, 'Ờ': 6, 'Ở': 6, 'Ỡ': 6, 'Ợ': 6,

        # U and variants
        'Ú': 3, 'Ù': 3, 'Ủ': 3, 'Ũ': 3, 'Ụ': 3,
        'Ư': 3, 'Ứ': 3, 'Ừ': 3, 'Ử': 3, 'Ữ': 3, 'Ự': 3,

        # Y variants
        'Ý': 7, 'Ỳ': 7, 'Ỷ': 7, 'Ỹ': 7, 'Ỵ': 7
    }
    
    # Master numbers
    MASTER_NUMBERS = {11, 22, 33}

    # Karmic debt numbers
    KARMIC_NUMBERS = {13, 14, 16, 19}

    def __init__(self, dob: str, name: str, current_date: str = None):
        """
        Initialize TBI Calculator with name, birthday and current date similar to numerology calculation
        
        Args:
            dob: Birthday (dd/mm/yyyy)
            name: Full name
            current_date: Current date (dd/mm/yyyy), default is today similar to numerology calculation
        """
        self.dob = dob
        self.name = name.upper().strip()
        self.current_datetime = current_date
        
        # Parse dates
        self.dob_date = self._parse_date(dob, "birthday")
        
        if self.current_datetime is None:
            vntz = pytz.timezone("Asia/Ho_Chi_Minh")
            self.current_datetime = datetime.now(vntz)
        else:
            self.current_datetime = self._parse_date(current_date, "current")
        
        # Convert name to numbers
        self.name_numbers = self._name_to_numbers()
        
        # Parse date components
        self.dob_day = self.dob_date.day
        self.dob_month = self.dob_date.month
        self.dob_year = self.dob_date.year
        
        # Calculate reduced date components
        self.day_r = self.reduce_number_with_masters(self.dob_day)
        self.month_r = self.reduce_number_with_masters(self.dob_month)
        self.year_r = self.reduce_number_with_masters(self.dob_year)

        # Calculate date no master
        self.day_r_no_master = self.reduce_number_no_master(self.dob_day)
        self.month_r_no_master = self.reduce_number_no_master(self.dob_month)
        self.year_r_no_master = self.reduce_number_no_master(self.dob_year)

    def _parse_date(self, date_str: str, date_type: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            return datetime.strptime(date_str, '%d/%m/%Y')
        except ValueError:
            if date_type == "current date":
                # Use current time if current date is invalid
                vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
                return datetime.now(vn_timezone)
            else:
                raise ValueError(f"Invalid {date_type} format. Use 'dd/mm/yyyy' format.")

    def _name_to_numbers(self) -> List[int]:
        """Convert name to list of numbers using ALPHABET mapping."""
        # Remove spaces and convert to uppercase
        clean_name = ''.join(self.name.upper().split())
        return [self.ALPHABET.get(char, 0) for char in clean_name if char in self.ALPHABET]

    def reduce_number(self, n: int) -> int:
        """
        Reduce number to single digit or master number (11, 22).

        Args:
            n: Number to reduce

        Returns:
            Reduced number (1-9, 11, or 22)
        """
        while n > 9 and n not in {11, 22}:
            n = sum(int(digit) for digit in str(n))
        return n
    
    def reduce_number_no_master(self, n: int) -> int:
        """
        Reduce number to single digit. (1-9)
        """
        while n > 9:
            n = sum(int(digit) for digit in str(n))
        return n

    def reduce_number_with_masters(self, n: int, masters: Set[int] = None) -> int:
        """
        Reduce number keeping master numbers (11, 22, 33).

        Args:
            n: Number to reduce
            masters: Set of master numbers to preserve (default: {11, 22, 33})

        Returns:
            Reduced number (1-9, 11, 22, or 33)
        """
        if masters is None:
            masters = self.MASTER_NUMBERS

        while n > 9 and n not in masters:
            n = sum(int(digit) for digit in str(n))
        return n

    def reduce_to_single_digit(self, n: int) -> int:
        """
        Always reduce to single digit (1-9).

        Args:
            n: Number to reduce

        Returns:
            Single digit (1-9)
        """
        while n > 9:
            n = sum(int(digit) for digit in str(n))
        return n

    def _is_vowel(self, char: str, current_word: str = None) -> bool:
        """
        Check if character is a vowel for soul number calculation.

        Rules:
        1. A, E, I, O, U, Y are vowels
        2. Variants with diacritics (Â, Ă, Ê, Ô, Ơ, etc.) are vowels
        3. Y is only a vowel when:
           - It's the only vowel in the word, OR
           - It stands alone
        4. All other cases with Y are consonants
        """
        char_upper = char.upper()

        # Basic vowels: A, E, I, O, U
        basic_vowels = {
            'A', 'E', 'I', 'O', 'U',
            # A variants with diacritics
            'Á', 'À', 'Ả', 'Ã', 'Ạ',
            'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ',
            'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ',
            'Ă', 'Â',
            # E variants with diacritics
            'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ',
            'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ',
            'Ê',
            # I variants with diacritics
            'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị',
            # O variants with diacritics
            'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ',
            'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ',
            'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ',
            'Ô', 'Ơ',
            # U variants with diacritics
            'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ',
            'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự',
            'Ư'
        }

        # Check if it's a basic vowel
        if char_upper in basic_vowels:
            return True

        # Special handling for Y
        if char_upper == 'Y' or char_upper in {'Ý', 'Ỳ', 'Ỷ', 'Ỹ', 'Ỵ'}:
            if current_word:
                return self._is_y_vowel_in_word(char, current_word)
            else:
                # Fallback: find the word containing this Y character
                word = self._find_word_with_char_at_position(char)
                return self._is_y_vowel_in_word(char, word)

        return False

    def _is_y_vowel_in_word(self, char: str, word: str) -> bool:
        """
        Check if Y is a vowel in the given word.

        Y is a vowel when:
        - It's the only vowel in the word, OR
        - It stands alone
        """
        if not word:
            return False

        # Count other vowels in the word (excluding this Y)
        other_vowels_count = 0
        for c in word:
            if c.upper() != 'Y':
                c_upper = c.upper()
                # Check if it's a vowel (A, E, I, O, U and their variants)
                if (c_upper in {'A', 'E', 'I', 'O', 'U', 'Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư'} or
                    c_upper in {'Á', 'À', 'Ả', 'Ã', 'Ạ', 'Ắ', 'Ằ', 'Ẳ', 'Ẵ', 'Ặ', 'Ấ', 'Ầ', 'Ẩ', 'Ẫ', 'Ậ',
                               'É', 'È', 'Ẻ', 'Ẽ', 'Ẹ', 'Ế', 'Ề', 'Ể', 'Ễ', 'Ệ',
                               'Í', 'Ì', 'Ỉ', 'Ĩ', 'Ị',
                               'Ó', 'Ò', 'Ỏ', 'Õ', 'Ọ', 'Ố', 'Ồ', 'Ổ', 'Ỗ', 'Ộ', 'Ớ', 'Ờ', 'Ở', 'Ỡ', 'Ợ',
                               'Ú', 'Ù', 'Ủ', 'Ũ', 'Ụ', 'Ứ', 'Ừ', 'Ử', 'Ữ', 'Ự'}):
                    other_vowels_count += 1

        # Y is vowel if it's the only vowel in the word
        return other_vowels_count == 0

    def _find_word_with_char_at_position(self, char: str) -> str:
        """
        Find the word containing the given character at its specific position.
        """
        name_parts = self._split_name_parts()

        # Find the position of this character in the original name
        char_pos = self.name.find(char)
        if char_pos == -1:
            return ""

        # Find which word contains this character at this position
        current_pos = 0
        for part in name_parts:
            part_start = current_pos
            part_end = current_pos + len(part)

            if part_start <= char_pos < part_end:
                return part

            current_pos = part_end + 1  # +1 for space

        return ""

    def _is_consonant(self, char: str, current_word: str = None) -> bool:
        """Check if character is a consonant."""
        return not self._is_vowel(char, current_word)

    def _split_name_parts(self) -> List[str]:
        """Split name into individual parts (words)."""
        return [part.strip() for part in self.name.split() if part.strip()]

    def calculate_ppa(self) -> int:
        """Calculate Path Potential Alignment
        
        Formula: reduce_number_with_masters(day + month + year) (keep master numbers)
        """
        return self.reduce_number_with_masters(self.day_r + self.month_r + self.year_r)

    def calculate_spi(self) -> int:
        """
        Calculate Skill Potential Index.

        Formula: reduce_number_with_masters(sum(nameNumbers)) (keep master numbers)
        """
        return self.reduce_number_with_masters(sum(self.name_numbers))


    def calculate_cmi(self) -> int:
        """
        Calculate Crisis Management Index.
        
        Formula: reduce_to_single_digit(sum(first_letters)) (always 1 digit)
        Get the first letter of each word in the full name, add all of them
        """
        name_parts = self._split_name_parts()
        if len(name_parts) < 1:  # Only need at least 1 word
            return 0
            
        first_letters_sum = 0
        for part in name_parts:
            if part:
                first_letter = part[0].upper()
                first_letters_sum += self.ALPHABET.get(first_letter, 0)
        
        return self.reduce_to_single_digit(first_letters_sum)

    def calculate_edi(self) -> int:
        """
        Calculate Emotional Drive Index.

        Formula: reduce_number_with_masters(sum(reduce_number_with_masters(sum(vowels(part))) for part in parts))
        """
        name_parts = self._split_name_parts()
        soul_sum = 0

        for part in name_parts:
            part_vowels_sum = sum(
                self.ALPHABET.get(char.upper(), 0)
                for char in part
                if self._is_vowel(char, part)
            )
            soul_sum += self.reduce_number_with_masters(part_vowels_sum)

        return self.reduce_number_with_masters(soul_sum)

    def calculate_mpi(self) -> int:
        """
        Calculate Market Persona Index.

        Formula: reduce_number_with_masters(sum(reduce_number_with_masters(sum(consonants(part))) for part in parts))
        """
        name_parts = self._split_name_parts()
        personality_sum = 0

        for part in name_parts:
            part_consonants_sum = sum(
                self.ALPHABET.get(char.upper(), 0)
                for char in part
                if self._is_consonant(char)
            )
            personality_sum += self.reduce_number_with_masters(part_consonants_sum)

        return self.reduce_number_with_masters(personality_sum)

    def calculate_nei(self) -> int:
        """
        Calculate Natural Edge Index.

        Formula: reduce_number(day) (keep master number 11/22)
        """
        return self.reduce_number_with_masters(self.dob_day)

    def calculate_ssi(self) -> int:
        """
        Calculate Subconscious Stability Index.

        Formula: 9 - count(missing_aspects)
        (missing_aspects are numbers 1..9 not in nameNumbers)
        """
        missing_aspects = self.get_wmi()
        return 9 - len(missing_aspects)

    def calculate_ri(self) -> int:
        """
        Calculate Resilience Index.

        Formula: reduce_number(life_path + life_purpose) (keep master number 11/22)
        """
        life_path = self.calculate_ppa()
        life_purpose = self.calculate_spi()
        return self.reduce_number_with_masters(life_path + life_purpose)

    def get_wmi(self) -> Set[int]:
        """
        Get Weakness Map Index.

        Returns:
            Set of numbers 1-9 not appearing in name_numbers
        """
        name_digits = set()
        for num in self.name_numbers:
            for digit in str(num):
                if digit.isdigit():
                    name_digits.add(int(digit))

        return sorted(set(range(1, 10)) - name_digits)

    def check_bli(self) -> str:
        """
        Check for Behavioral Liability Index.

        Formula:
        - karmic = {13, 14, 16, 19}
        - dobSum = sum(all digits in day, month, year)
        - If dobSum ∈ karmic or sum(nameNumbers) ∈ karmic ⇒ "Has Karmic Debt"
        """
        # Sum of all digits in date of birth
        dob_sum = sum(int(digit) for digit in f"{self.dob_day}{self.dob_month}{self.dob_year}")

        # Sum of name numbers
        name_sum = sum(self.name_numbers)

        if dob_sum in self.KARMIC_NUMBERS or name_sum in self.KARMIC_NUMBERS:
            return "Có Karmic Debt"
        return "Không có Karmic Debt"

    def calculate_sai(self) -> List[int]:
        """
        Calculate Strength Amplifier Index.

        Formula: Count frequency of numbers in nameNumbers, get the most frequent numbers
        """
        from collections import Counter

        digit_counts = Counter()
        for num in self.name_numbers:
            for digit in str(num):
                if digit.isdigit():
                    digit_counts[int(digit)] += 1

        if not digit_counts:
            return []

        max_freq = max(digit_counts.values())
        return sorted([num for num, freq in digit_counts.items() if freq == max_freq])
    
    def get_societal_adaptability_index(self) -> str:
        """
        Get societal adaptability index based on birth year.

        Formula:
        - 1981–1996: "Gen Y (Millennials) - Cân bằng công việc-cuộc sống, công nghệ"
        - 1997–2012: "Gen Z - Công nghệ số, đa dạng, thay đổi nhanh"
        - Khác: "Khác"
        """
        if 1981 <= self.dob_year <= 1996:
            return "Gen Y (Millennials) - Cân bằng công việc-cuộc sống, công nghệ"
        elif 1997 <= self.dob_year <= 2012:
            return "Gen Z - Công nghệ số, đa dạng, thay đổi nhanh"
        else:
            return "Khác"

    def calculate_ppai(self) -> int:
        """
        Calculate Path–Potential Alignment Index.

        Formula: reduceNumber(abs(soul - personality)) (Keep master number 11/22)
        """
        lifepath = self.calculate_ppa()
        life_purpose = self.calculate_spi()
        return self.reduce_number(abs(lifepath - life_purpose))

    def calculate_ioci(self) -> int:
        """
        Calculate Inner–Outer Coherence Index.

        Formula: reduceToSingleDigit(abs(soul - personality)) (1 digit)
        """
        soul = self.calculate_edi()
        if soul in [11, 22, 33]:
            soul = sum(int(digit) for digit in str(soul))
        personality = self.calculate_mpi()
        return self.reduce_to_single_digit(abs(soul - personality))
    
    def calculate_tci_phase(self) -> Dict[str, int]:
        """
        Calculate Trading Cycle Index Phase.

        Formula (keep master number 11/22 at each reduceNumber step):
        - dayM = reduceNumber(day); monthM = reduceNumber(month); yearM = reduceNumber(year)
        - tci_1 = reduceNumber(monthM + dayM)     // Month + Day
        - tci_2 = reduceNumber(dayM + yearM)      // Day + Year
        - tci_3 = reduceNumber(tci_1 + tci_2)
        - tci_4 = reduceNumber(monthM + yearM)    // Month + Year
        """
        day_m = self.reduce_number(self.dob_day)
        month_m = self.reduce_number(self.dob_month)
        year_m = self.reduce_number(self.dob_year)

        tci_1 = self.reduce_number(month_m + day_m)
        tci_2 = self.reduce_number(day_m + year_m)
        tci_3 = self.reduce_number(tci_1 + tci_2)
        tci_4 = self.reduce_number(month_m + year_m)

        return {
            "tci_1": tci_1,
            "tci_2": tci_2,
            "tci_3": tci_3,
            "tci_4": tci_4
        }
        
    def calculate_cii(self) -> int:
        """
        Calculate Cohort Influence Index.
        
        Formula: reduceNumber(year) (keep master number 11/22/33)
        """
        generation = sum(int(digit) for digit in str(self.dob_year))
        return self.reduce_number_with_masters(generation)

    def calculate_tai(self) -> int:
        """
        Calculate Trading Attitude Index.
        
        Formula: reduceNumber(sum(int(digit) for digit in str(self.dob_day + self.dob_month))) (always 1 digit)
        """
        sum_day = sum(int(digit) for digit in str(self.dob_day))
        sum_month = sum(int(digit) for digit in str(self.dob_month))
        return self.reduce_number_no_master(sum_day + sum_month)

    def calculate_bci(self) -> Dict[str, int]:
        """
        Calculate Behavioral Challenge Index.

        Formula (keep dayR, monthR, yearR like life_path - keep master number 11/22/33):
        - bci_1 = abs(dayR - monthR)
        - bci_2 = abs(dayR - yearR)
        - bci_3 = abs(bci_1 - bci_2)
        - bci_4 = abs(monthR - yearR)
        """
        bci_1 = abs(self.day_r_no_master - self.month_r_no_master)
        bci_2 = abs(self.day_r_no_master - self.year_r_no_master)
        bci_3 = abs(bci_1 - bci_2)
        bci_4 = abs(self.month_r_no_master - self.year_r_no_master)

        return {
            "bci_1": bci_1,
            "bci_2": bci_2,
            "bci_3": bci_3,
            "bci_4": bci_4
        }
        
    def calculate_ari(self) -> int:
        """
        Calculate Analytical Reasoning Index.
        
        Formula: reduceNumberWithMasters(day + sum(letters_of_given_name)) (keep master number 11/22/33)
        Get the given name (last part in full name), add the value of each letter + day of birth (day)
        """
        name_parts = self._split_name_parts()
        if not name_parts:
            return 0
            
        given_name = name_parts[-1]  # Last part (given name)
        given_name_sum = sum(
            self.ALPHABET.get(char.upper(), 0)
            for char in given_name
            if char.upper() in self.ALPHABET
        )
        
        return self.reduce_number_with_masters(self.dob_day + given_name_sum)

    def calculate_age_tci(self) -> List[int]:
        """
        Calculate Age Trading Cycle Index.

        Formula:
        - If life_path ∈ {11,22,33} ⇒ start = 36 - 4 = 32
        - Otherwise 36 - life_path
        - Array of 4 milestones: [start, start+9, start+18, start+27]
        """
        life_path = self.calculate_ppa()

        if life_path in self.MASTER_NUMBERS:
            start = 32  # 36 - 4
        else:
            start = 36 - life_path

        return [start, start + 9, start + 18, start + 27]
    
    def calculate_alignment_signals(self) -> Dict[str, int]:
        """
        Calculate Alignment Signals (AMI, MRI and DAI).

        Formula (with currentDate = dd/mm/yyyy):
        - ami = reduceNumber(day + month + currentYear)      // keep master number 11/22
        - mri = reduceNumber(currentMonth + personal_year)
        - dai = reduceNumber(currentDay + currentMonth + personal_year)
        """
        current_year = self.current_datetime.year
        current_month = self.current_datetime.month
        current_day = self.current_datetime.day

        # Annual Momentum Index (AMI)
        personal_year = self.dob_day + self.dob_month + current_year

        if current_month < self.dob_month or (current_month == self.dob_month and current_day < self.dob_day):
            personal_year -= 1
        
        personal_year = self.reduce_number_no_master(personal_year)

        # Monthly Rhythm Index (MRI)
        personal_month = self.reduce_number_no_master(current_month + personal_year)

        # Daily Alignment Index (DAI)
        personal_day = self.reduce_number_no_master(current_day + current_month + personal_year)

        return {
            "ami": personal_year,
            "mri": personal_month,
            "dai": personal_day
        }

    def get_all_tbi_indicators(self) -> Dict[str, Union[int, str, List[int]]]:
        """Calculate all TBI indicators similar to numerology calculation"""
        return {
            "day_of_birth": self.dob_date.strftime('%d/%m/%Y'),
            "current_date": self.current_datetime.strftime('%d/%m/%Y'),
            "edi": self.calculate_edi(),
            "ppai": self.calculate_ppai(),
            "spi": self.calculate_spi(),
            "cmi": self.calculate_cmi(),
            "mpi": self.calculate_mpi(),
            "ri": self.calculate_ri(),
            "ioci": self.calculate_ioci(),
            "tai": self.calculate_tai(),
            "ppa": self.calculate_ppa(),
            "wmi": self.get_wmi(),
            "ssi": self.calculate_ssi(),
            "sai": self.calculate_sai(),
            "bci": self.calculate_bci(),
            "nei": self.calculate_nei(),
            "bli": self.check_bli(),
            "ari": self.calculate_ari(),
            "tci": self.calculate_tci_phase(),
            "cii": self.calculate_cii(),
            "alignment_signals": self.calculate_alignment_signals(),
            "age_tci": self.calculate_age_tci()
        }

    def get_tbi_summary(self) -> Dict[str, Any]:
        """Get summary of TBI indicators"""
        indicators = self.get_all_tbi_indicators()
        
        return {
            "user_info": {
                "name": self.name,
                "birthday": self.dob_date,
                "current_date": self.current_datetime.strftime('%d/%m/%Y')
            },
            "tbi_indicators": indicators,
            "core_indicators": {
                "ppa": indicators["ppa"],
                "spi": indicators["spi"],
                "edi": indicators["edi"],
                "ppai": indicators["ppai"],
            },
            "behavioral_emotional_indicators": {
                "ri": indicators["ri"],
                "mpi": indicators["mpi"],
                "ioci": indicators["ioci"],
                "tai": indicators["tai"],
                "cmi": indicators["cmi"],
            },
            "thinking_decision_making_indicators": {
                "wmi": indicators["wmi"],
                "ssi": indicators["ssi"],
                "sai": indicators["sai"],
                "nei": indicators["nei"],
                "bli": indicators["bli"],
                "ari": indicators["ari"],
            },
            "timing_indicators": {
                "tci": indicators["tci"],
                "bci": indicators["bci"],
                "mri": indicators["alignment_signals"]["mri"],
                "dai": indicators["alignment_signals"]["dai"],
                "ami": indicators["alignment_signals"]["ami"],
                "cii": indicators["cii"],
            }
        }


class TBICalculatorFactory:
    """Factory class to create TBI Calculator"""
    
    @staticmethod
    def create_calculator(dob: str, name: str, current_date: str = None) -> TBICalculator:
        """Create TBI Calculator instance"""
        return TBICalculator(dob, name, current_date)
    
    @staticmethod
    def create_calculator_for_today(dob: str, name: str) -> TBICalculator:
        """Calculate TBI Calculator with current date"""
        return TBICalculator(dob, name)
