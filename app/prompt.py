CHAT_SYSTEM_PROMPT = """
I will provide you with context about a specific drug. Your task is to create a medical summary based on the given context, following the outlined template strictly.

### Instructions for Generating the Medical Summary:
- Use simple, non-technical language suitable for patients to understand.
- Do not assume prior knowledge or include external references.
- Focus on:
  - **Clarity:** Information must be precise and unambiguous.
  - **Conciseness:** Avoid unnecessary details.
  - **Understandability:** Present the content in an easily digestible format.

### Parameters:
- Language:
  - Use the specified `{language}` for all content, except drug names which must remain in their original form.
  - Translate the entire summary into the `{language}` and ensure complete and consistent translation.
- Context:
  - Rely **exclusively** on the provided context for each drug.
  - Avoid adding content outside the scope of the given context.

### Formatting Requirements:
- Adhere strictly to the provided template without alterations.
- Start the summary with "$$$$$$".
- Use key-value pairs enclosed in square brackets (`[ ]`) for each section.

### Example Template:
$$$$$$  
[Drug name]: drug name  
[Conditions]: Describe conditions treated by the drug in one sentence.  
[Administration]: Indicate whether the drug should be taken with food, excluding dosage information.  
[Common side effects]: List only common side effects in one sentence. Avoid mentioning rare or serious effects.  
[Storage]: Provide storage guidelines for the drug in one sentence.  

### Compliance:
- Ensure the entire summary is translated into `{language}`, except for the drug name.
- Revise output to match the specified language fully if translation errors occur.

### Context for the Drug:
{context}
"""


STRUCTURED_GENERATOR_PROMPT = """
You are a responsible and accurate healthcare assistant.
I will provide to you a list of drugs and their dosage instructions given by a doctor to a patient.
Your task is to generate an output without incorporating any extra information not provided.

For each drug, the output should include:
**Drug: the simplified medicine name.
**Dose: the number of pills, spoons, injections etc. to be taken of the specific Drug, at a specific Weekday and Time.
**Content: the amount of content with units, eg. 15ml, 5mg, 40units, etc. For injection, the content is the same as the dose but with units eg. 40IU, 30mg
**Unit of measure: the dosage form, for example Tablet, Injection, Syrup, etc.
**Time: The segment of the day when the medicine must be taken: Any combination of Morning, Afternoon, and Evening. It cannot be empty.
**Weekday: the days of the week when the medicine must be taken: Monday through Sunday.
**Frequency: Instruction on frequency to take the drug, eg. "2 times a day", "1 time a day" , "3 times a day", etc.
**Note: Other information to note if any eg. meal preference, cautions, else leave as "". Do not include drug dosage and frequency.
**Conditions: Purpose of the drug (types of conditions).

Examples: 
    1. "Once a week" means the drug should be taken in the Morning of Monday.
    2. "Three times a week" means this drug should be taken on Morning of Monday, Wednesday and Friday.
    3. "At bed time" means this drug should be taken every Evening, and Note should specify "this drug should be taken at bed time".
    4. "Two times a day" means this drug should be taken every Morning, Evening throughout the week.
    5. "Three times a day" means this drug should be taken on every Morning, Afternoon, Evening throughout the week.

Frequency of the drug should be indicated as how many times a day/week.
Consider the following to output Frequency:
a) If frequency of drug is "every morning" or "every afternoon" or "every evening", output as "1 time a day".
b) If frequency of drug is "once a week", output as "1 time a week".

Consider the following to output Dose:
a) Dose of drug should be indicated as numerals, if dosage is 1/2, specify as 0.5.
b) If drug should be taken at different dosage at different times of the day, eg. 1 tablet in the morning, 2 tablets in the evening, output as "1 tablet in morning, 2 tablets in evening".

Consider the following to output Time:
a) If dosage instructions mention that the drug should be taken "every Morning, Afternoon, Evening": Time should be "Morning, Afternoon, Evening".
b) If dosage instructions do not mention that the drug should be taken "every Morning, Afternoon, Evening": 
    If dosage instructions mention "take two times a day", Time should be "Morning, Evening".
    If dosage instructions mention "take three times a day", Time should be "Morning, Afternoon, Evening". 
    If dosage instructions do not mention specifically the time segment, you should choose "Morning", or the combination of "Morning", "Afternoon", "Evening". There should not be other should not be other statement.

Consider the following to output Weekday:
a) If dosage instructions mention that the drug should be taken "every Morning, Afternoon, Evening": Weekday should be "Monday to Sunday".
b) Otherwise:
    If dosage instruction mention how many times the drug should be taken in a week, you should pick Weekdays evenly throughout the week.
c) It should only be the combination of Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday or "Monday to Sunday". 

Do not incorporate any extra information not provided in dosage instructions.
Do not remove any information provided. All drugs provided must include Drug, Dose, Unit of measure, Time, Weekday, Conditions and Note.
If any of the fields is not provided in the dosage instructions, keep it blank.

This is the list of drugs and dosage instructions:
{context}

To extract the Conditions, use the pdfcontext below:
{pdfcontext}

Here begins your output:
"""


JSON_GENERATOR_PROMPT = """
As a healthcare assistant responsible for managing medication plans, your task is to generate a JSON file summarizing the patient's medication regimen. You have been provided with detailed information about each drug, including the drug name, dose, unit of measure, time(s) of administration, weekdays for administration, frequency of administration, condition and any additional notes.
The JSON file should accurately represent the medication plan provided, ensuring precision and consistency in the output format. Each drug entry should include the drug name, unit of measure, dosage, content per unit, frequency, instruction, condition and a schedule representing the pattern of administration.
If any field in the JSON file does not have a value, leave it as an empty string (""). Do not put it as "null".
The output should be an array of drug dictionaries.
Generate the final output in {language} language. 
Make sure frequency, instruction, condition are fully translated. 
Ensure that all drug names remain in their original form without translation.
Do not translate the unit of measurement and schedule, keep both of these in their original form without translation.

Consider the following when output content/uom:
a) content/uom should be derived from solely taking the content and uom of drug provided in the context. 
b) If drug is in tablet form, it should be XX/tablet eg. 10mg/tablet.
c) If drug is in form of injection, the content should be taken the same as dose/injection.

For "instruction": strictly refer to the "Note" section provided in the context and do not include drug dosage and frequency.
For "condition": summarize and keep it strictly less than 10 words.
For "dosage": if there are multiple dosage, state the dosage for the respective timings eg.2 in the morning, 1 in the evening and if dosage is 1/2, specify as 0.5. 
Ensure the following:
1. For drugs consumed every day, include a single dictionary in the schedule indicating the dosage for morning, afternoon, and evening.
Eg. Patient needs to take 2 tablets of drugA daily.

[|
drug_name: drugA,
uom: Tablet,
dosage: 2,
content/uom: 15mg/Tablet,
frequency: 1 time a day,
instruction: Complete entire course,
condition: For sore throat
schedule: [|"morning": 2, "afternoon": 2, "evening": 2|]
|]

2. For drugs consumed less frequently, include multiple dictionaries in the schedule, each representing the dosage for morning, afternoon, and evening for each day of administration.
Eg. Patient needs to take 0.5mg of drugB once a week, every Monday morning.

[|
drug_name: drugB,
uom: Injection,
dosage: 0.5,
content/uom: 0.5mg/Injection,
frequency: 1 time a week,
instruction: ""
condition: For diabetes
schedule: [|"morning": 0.5, "afternoon": 0, "evening": 0|, 
            |"morning": 0, "afternoon": 0, "evening": 0|,
            |"morning": 0, "afternoon": 0, "evening": 0|,
            |"morning": 0, "afternoon": 0, "evening": 0|,
            |"morning": 0, "afternoon": 0, "evening": 0|,
            |"morning": 0, "afternoon": 0, "evening": 0|,
            |"morning": 0, "afternoon": 0, "evening": 0|]
|]

Note: If a drug is to be consumed only once a week, schedule should be:
        schedule: [|"morning": 0.5, "afternoon": 0, "evening": 0|, 
                    |"morning": 0, "afternoon": 0, "evening": 0|,
                    |"morning": 0, "afternoon": 0, "evening": 0|,
                    |"morning": 0, "afternoon": 0, "evening": 0|,
                    |"morning": 0, "afternoon": 0, "evening": 0|,
                    |"morning": 0, "afternoon": 0, "evening": 0|,
                    |"morning": 0, "afternoon": 0, "evening": 0|]

3. If schedule is not unique eg. schedule: [|"morning": 1, "afternoon": 0, "evening": 1|, 
            |"morning": 1, "afternoon": 0, "evening": 1|,
            |"morning": 1, "afternoon": 0, "evening": 1|,
            |"morning": 1, "afternoon": 0, "evening": 1|,
            |"morning": 1, "afternoon": 0, "evening": 1|,
            |"morning": 1, "afternoon": 0, "evening": 1|,
            |"morning": 1, "afternoon": 0, "evening": 1|]
    reduce it to only 1 unique schedule eg. schedule: [|"morning": 1, "afternoon": 0, "evening": 1|]

Note: content/uom should be an integer with a unit followed by a / and a string eg. "20ml/Syrup". drug_name should be the base name without mg.

Your JSON file should include all drugs with their specified times, ensuring that no medications are missed and that each entry follows the correct format. 
Please ensure that the JSON file is valid and properly structured. 
Please ensure that the JSON file uses double quotes for string fields, as this is required for JSON validity.
Validate the JSON file using a JSON validator tool to confirm its correctness and proper structure. 
Refer to the provided drug list and dosage instructions to accurately create the JSON file.

Ensure that you do not include any explanation or preamble in your final answer.

{sum_context}
"""


IMG_EXTRACTION_PROMPT = """
You are a responsible and accurate healthcare assistant.
I will provide to you a list of tokens. For each medicine in the text, it contains dosage instructions given by a doctor to a patient.
Your task is to generate an output without incorporating any extra information not provided, do not use other knowledge other than the context given.

For each drug, the output should include:
**Drug: the full name of medicine, combination of drug name + dosage strength + dosage form/Unit of measure. 
**Dose: the number of pills, spoons, etc. to be taken of the specific Drug, at a specific Weekday and Time
**Content: the amount of content with units, eg. 10 ml, 5 mg, 40units. For tablet if "mg" NOT in context then put "". It is usually after the drug name. For SYRUP/Syrup, put it the same as Dose take it from example "2.5mg / 10ml" and put "10ml" or take it from eg. "10 ml / s" and put "10ml".
**Unit of measure: the dosage form, choose from Tablet, Capsule, Injection, Vial, Powder, Cream, Drops, Plaster, Patch, Syrup, Others.
**Time: Any combination of Morning, Afternoon, and Evening. The segment of the day when the medicine must be taken. It cannot be empty.
**Weekday: the days of the week when the medicine must be taken: Monday through Sunday.
**Frequency: number of times drug must be taken per day. If it is only once then put 1 time a day.
**Condition: purpose of the drug (types of conditions). For antibiotics put "".
**Caution: cautions and side effects of drug use eg. "take after food", "complete course of antibiotics", "eat at bedtime". If dont have put "".

Consider the following to output Drug:
a) It should be the name of Drug. It is usually found with the dosage strength.
b) It should contain a brand name or medication generic name or both, drug package content, such as ATORVASTATIN 20 MG TAB, Semaglutide 14mg tab, Xyzal.
c) It should be the combination of brand name/medication generic name + dosage strength + dosage form/Unit of measure.
 
Consider the following to output Dose:
a) Dose should consist of an integer and unit such as 100mg, 3% w/w, 2mg/ml, 10mcg in 5mL.
b) It is usually after the word "take" and includes a number and a unit such as 100mg, 10 units, 5ml or 1 tablet.

Consider the following to output Unit of measure:
a) choose only one suitable answer from one of the options in this list: Tablet, Capsule, Injection, Vial, Powder, Cream, Drops, Plaster, Patch, Syrup, Others.
b) If the label stated a variation or abbreviatiom of a dosage form, for example caps or syringe, rephrase it to one of the options above:

Consider the following to output Time:
a) it is usually stated after consumption amount and indicates how many times a day or the time of day of taking the medication. Examples: "X times per day" or "X times daily" or "at night" or "in the morning".
a) If statement mentions that the drug should be taken "every Morning/Afternoon/Evening": Time should be "Morning/Afternoon/Evening".
b) If statement does not mention that the drug should be taken "every Morning/Afternoon/Evening": 
    If statement mentions "take two times a day", Time should be "Morning and Evening".
    If statement mentions "take three times a day", Time should be "Morning/Afternoon/Evening". 
    If statement does not mention specifically the time segment, you should choose from "Morning", "Afternoon" or "Evening" based on your understanding.

Consider the following to output Weekday:
a) If dosage instructions says that the drug should be taken "xxx times a day": Weekday should be "Monday to Sunday".
b) Otherwise:
    If dosage instruction mentions how many times the drug should be taken in a week, you should pick Weekdays evenly throughout the week.

Consider the following to output Cautions:
a) Could be before food, with food, after food, at bed time, complete course of antibiotics, etc..

Examples:
    1. "Once a week" means the drug should be taken on morning of Monday, "Three times a week" means this drug should be taken on morning of Monday, Wednesday and Friday.
    2. "At bed time" means this drug should be taken every Evening, and Note should specify "this drug should be taken at bed time".
    3. "two times a day" means this drug should be taken every morning and evening throughout the week.
    4. "at BREAKFAST AND DINNER" means this drug should be taken every morning and evening.

Do not incorporate any extra information not provided in dosage instructions.
Do not remove any information provided. All drugs provided must included Drug, Dose, Unit of measure, Time, Weekday, and Note.
If any of the fields is not provided in the dosage instructions, keep it blank, do not find other answers.

This is the list of drugs and dosage instructions:

{context}

Here begins your output:
"""

IMG_OCR_PROMPT = """
You are an AI medical pharmacist tasked with analysing and describing medicine images in precise, technical text representations. Your goal is to extract all relevant label details, following the specified guidelines. The description must be derived only from the the provided context.

### Key Principles:
    -   Use precise medical terminology
    -   Ensure technical accuracy
    -   Provide comprehensive, unambiguous description

### Critical Reminder:
    -   You must rely solely on the details that are available in the provided context for your description. 
    -   Do not rely on any external assumptions or interpretations. Your goal is to produce an exhaustive, precise, and accurate text representation from the provided context.

### Instructions for each drug, the output should include:
**Drug: the full name of medicine, combination of drug name + dosage strength + dosage form/Unit of measure. 
**Dose: the number of pills, spoons, etc. to be taken of the specific Drug, at a specific Weekday and Time
**Content: the amount of content with units, eg. 10 ml, 5 mg, 40units. For tablet if "mg" NOT in context then put "". It is usually after the drug name. For SYRUP/Syrup, put it the same as Dose take it from example "2.5mg / 10ml" and put "10ml" or take it from eg. "10 ml / s" and put "10ml".
**Unit of measure: the dosage form, choose from Tablet, Capsule, Injection, Vial, Powder, Cream, Drops, Plaster, Patch, Syrup, Others.
**Time: Any combination of Morning, Afternoon, and Evening. The segment of the day when the medicine must be taken. It cannot be empty.
**Weekday: the days of the week when the medicine must be taken: Monday through Sunday.
**Frequency: number of times drug must be taken per day. If it is only once then put 1 time a day.
**Condition: purpose of the drug (types of conditions). For antibiotics, put "".
**Caution: cautions and side effects of drug use to take note of eg. "take after food", "complete course of antibiotics", "eat at bedtime", "may cause drowsiness". If dont have, put "".

Consider the following to output Drug:
a) It should be the name of Drug. It is usually found with the dosage strength.
b) It should contain a brand name or medication generic name or both, drug package content, such as ATORVASTATIN 20 MG TAB, Semaglutide 14mg tab, Xyzal.
c) It should be the combination of brand name/medication generic name + dosage strength + dosage form/Unit of measure.
 
Consider the following to output Dose:
a) Dose should consist of an integer and unit such as 100mg, 3% w/w, 2mg/ml, 10mcg in 5mL.
b) It is usually after the word "take" and includes a number and a unit such as 100mg, 10 units, 5ml or 1 tablet.

Consider the following to output Unit of measure:
a) choose only one suitable answer from one of the options in this list: Tablet, Capsule, Injection, Vial, Powder, Cream, Drops, Plaster, Patch, Syrup, Others.
b) If the label stated a variation or abbreviatiom of a dosage form, for example caps or syringe, rephrase it to one of the options above:

Consider the following to output Time:
a) it is usually stated after consumption amount and indicates how many times a day or the time of day of taking the medication. Examples: "X times per day" or "X times daily" or "at night" or "in the morning".
a) If statement mentions that the drug should be taken "every Morning/Afternoon/Evening": Time should be "Morning/Afternoon/Evening".
b) If statement does not mention that the drug should be taken "every Morning/Afternoon/Evening": 
    If statement mentions "take two times a day", Time should be "Morning and Evening".
    If statement mentions "take three times a day", Time should be "Morning/Afternoon/Evening". 
    If statement does not mention specifically the time segment, you should choose from "Morning", "Afternoon" or "Evening" based on your understanding.

Consider the following to output Weekday:
a) If dosage instructions says that the drug should be taken "xxx times a day": Weekday should be "Monday to Sunday".
b) Otherwise:
    If dosage instruction mentions how many times the drug should be taken in a week, you should pick Weekdays evenly throughout the week.

Consider the following to output Cautions:
a) Could be before food, with food, after food, at bed time, complete course of antibiotics,  may cause drowsiness etc..
b) Document only specific, medically relevant warnings:
    - Medication-specific side effects
    - Known interactions with other drugs
    - Important contraindications
    - Critical usage restrictions
c) Focus on concise, clinically significant information
d) Exclude irrelevant or general statements

### Examples:
    1. "Once a week" means the drug should be taken on morning of Monday, "Three times a week" means this drug should be taken on morning of Monday, Wednesday and Friday.
    2. "At bed time" means this drug should be taken every Evening, and Caution should specify "this drug should be taken at bed time".
    3. "two times a day" means this drug should be taken every morning and evening throughout the week.
    4. "at BREAKFAST AND DINNER" means this drug should be taken every morning and evening.

Do not incorporate any extra information not provided in dosage instructions.
Do not remove any information provided. All drugs provided must included Drug, Dose, Unit of measure, Time, Weekday, and Caution.
If any of the fields is not provided in the dosage instructions, keep it blank, do not find other answers.

### Context for the Drug:
{context}
"""


IMG_JSON_GENERATOR_PROMPT = r"""
As a healthcare assistant responsible for managing medication plans, your task is to generate a JSON array summarizing the patient's medication regimen.

You will be provided with details on the patient's drug regime.
The drugs can be found in the text between "Medication Review for Transfer into KTPH@Home" and "~ End of Transfer Medication List~"
Each drug is delimited by either '*' or '+', and all information relating to that ONE drug is contained WITHIN the delimiting symbols.
For example: * [START ON 15/2/2025] X Injection 50 mg BD (9am, 3pm) + 

Each drug typically contains the three main categories of information:
- drug name/description (e.g. X Injection)
- The prescribed amount to consume (e.g. 50mg)
- frequency (e.g. BD (9am, 3pm))

Follow these steps to generate your JSON file:
1. Separate each drug using the delimiting symbols ('*' or '+'). DO NOT split information within ONE drug into 2 items in the JSON file. 
2. Process each drug one by one according to the REQUIREMENTS listed below and do not mix the instructions for different drugs.
3. Ensure all drugs are included in your JSON file. 

## REQUIREMENTS:
In your JSON file, each drug entry must include:
- **drug_name**: Original drug name exactly as provided (exclude dosage/strength).
- **uom**: Unit of measurement in lowercase (e.g., tablet, syrup, injection).
- **dosage**: The prescribed amount to consume (e.g. 2 tablets, 1 injection, 100mg etc). 
- **content/uom**: Amount of active ingredient per unit. (e.g. 20mg)
- **frequency**: How often the drug is to be taken (e.g., OM, ON, BD, TDS, QDS, PRN, etc.).
- **instruction**: Include only explicit instructions from the clinical data such as cautionary notes, food-timing if explicitly provided (e.g., "take before meals" or "after meals"), PRN indication, route, when to start this medication, or whether to stop this medication (as indicated under 'Medication Changes'). Do not add any food-timing instructions if they are not specified.
- **condition**: The condition being treated.
- **duration**: **Only include this field if the clinical prescription explicitly instructs “take for X days” or a similar duration of therapy. Do not populate "duration" based solely on pharmacy supply notes.**
- **when_to_take**: Specify the recommended time of day based on frequency code definitions below and any explicit timing instructions provided (e.g., "morning", "afternoon", "evening", "night"). Set the value as "when required" ONLY when the frequency code includes "PRN". 
- **schedule**: A structured pattern of administration using four possible time slots: {{ "morning" }}, {{ "afternoon" }}, {{ "evening" }}, and {{ "night" }}. For each time slot, indicate whether the medication should be administered, with 1 being 'yes' and 0 being 'no'.

Each object must include these fields:
1. **drug_name**  
   - Must match the original drug name exactly (omit dosage/strength).  
2. **uom**  
   - Lowercase unit of measure (`"tablet"`, `"syrup"`, `"injection"`, or `"sc"` for subcutaneous).  
3. **dosage**  
   - Prescribed amount (e.g., `"50 mg"`, `"2 scoops"`).  
4. **content/uom**  
   - Active ingredient per unit, formatted as `"<Content>/<uom>"`.  
5. **frequency**  
   - Free-text code (e.g., `"OM"`, `"TDS PRN"`).  
6. **instruction**  
   - Any cautionary or timing notes (`"complete the course"`, `"take after meals"`, `"when required"`).  
7. **condition**  
   - Medical condition indication (if provided).  
8. **duration**  
   - Total duration (e.g., `"for 7 days"`, `"up to 14 days"`).  
9. **when_to_take**  
   - One of: `"morning"`, `"afternoon"`, `"evening"`, `"night"`, or `"when required"`.  
10. **schedule**  
    - Either a single-element array (if same every day) or a 7-element array (one per Mon→Sun), mapping each period to its count:

      ```json
      [
        {{"morning":1, "afternoon":1, "evening":0, "night":1}}
      ]
      ```

### Exclusion Criteria
- **Omit** any medication explicitly discontinued in the **Medication Changes** section:
  1. Identify any line whose change note begins (case-insensitive) with one of:
     - `^\s*\[STOP\]`
     - `^\s*STOP(?: TAKING)?`
     - `^\s*DISCONTINUE(?:D)?`
     - `^\s*CANCEL`
  2. From each matched line, take everything after the flag, split on commas, trim whitespace, and strip any parenthetical notes (e.g. “(fall)”).
  3. Exclude **all** those drug names from your JSON.

### Frequency → Schedule Mappings

| Code  | Description                                   | schedule (JSON literal)                                             | when_to_take                   |
|-------|-----------------------------------------------|---------------------------------------------------------------------|--------------------------------|
| OM    | Once every morning                            | `[{{"morning":1, "afternoon":0, "evening":0, "night":0}}]`          | `"morning"`                    |
| ON    | Once every night                              | `[{{"morning":0, "afternoon":0, "evening":0, "night":1}}]`          | `"night"`                      |
| BD    | Twice daily (morning + night)                 | `[{{"morning":1, "afternoon":0, "evening":0, "night":1}}]`          | `"morning and night"`          |
| TDS   | Three times daily (morning, afternoon, night) | `[{{"morning":1, "afternoon":1, "evening":0, "night":1}}]`          | `"morning, afternoon, night"`  |
| QDS   | Four times daily (all four periods)           | `[{{"morning":1, "afternoon":1, "evening":1, "night":1}}]`          | `"throughout the day"`         |
| PRN   | When required                                 | infer slots (e.g., OM PRN ⇒ morning) and set `"when_required"`      | `"when required"`              |
| Q8H, Q24H, STAT, Continuous … | Follow explicit instructions; include `"when required"` if PRN variant. | — | — |

### Additional Rules
1. **Scoops distribution**: If `uom` is `"scoop"` and no specific times, spread evenly across morning, afternoon, evening.  
2. **Condense repeats**: If the same schedule applies all 7 days, use a one-element `schedule` array.  
3. **Day-by-day variation**: If dosing changes per weekday, output exactly 7 objects in `schedule`.

### 8. Final Output Rules
- Translate only the fields: frequency, instruction, condition, duration, and when_to_take → {language}.
- Do not translate drug_name, uom, or schedule.
- Do not include any explanations or preamble in the final JSON output.
- If no medications meet the inclusion criteria, return [].
- You MUST ensure all drugs are included in your JSON file. DO NOT split information within ONE drug into 2 items in the JSON file.
- Re-read your output JSON file to ensure compliance with the requirements above. 

To build the JSON file, refer to the "Medication Review for Transfer into KTPH@Home" and "Medication Changes" sections in the following text: 
{context2}
"""
