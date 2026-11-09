# Showcase: the hardest questions MiniGPT v2.0-preview answers correctly

**MiniGPT v2.0-preview (98M). PREVIEW, NOT A RELEASE: failed 2 of 4 pre-registered bars (over-refusal 77/100, bare arithmetic 16/18). v1.0.1 remains the released model.**

Run on 10 November 2026. No training, and no sealed set was used or opened.

## In short

On 40 new questions of rising difficulty, **v2.0-preview answered 10 correctly (25%) and v1.0.1 answered 8 (20%)**,
marked strictly by hand. That difference is small: the models agree on 36 of the 40 questions (3 right only for
v2.0-preview, 1 only for v1.0.1), and one set of questions cannot separate them.

**v2.0-preview's correct answers are almost all short fixed phrases** learned in instruction tuning: "That didn't
happen.", "Albert Einstein never did that.", "I don't know. Nobody can predict that with certainty." and "I can't
know or share that." Of its 10 correct answers, 9 are such phrases. The tenth is a two-step proportion problem
(eggs for cakes) that follows a training layout.

Both models got every definition, "why" question and instruction-with-constraints question wrong except one
definition for v1.0.1. Only 10 of v2.0-preview's answers were correct, so its "10 hardest correct answers" are
simply all of them.

## How it was done

1. **The questions were written before any model was run.** There are 40, 10 per difficulty level (1 = easiest,
   4 = hardest), across six kinds: multi-step word problems, "why" science questions, definitions, false
   premises about real people, unknowable questions, and instructions with constraints. Each has an expected
   answer and a strict marking rule (`showcase/questions_v1.jsonl`, committed 10 Nov 2026 00:35 IST in the private development history).
2. **Contamination check** (`showcase/check_contamination.py`) against all training data of both models:
   - all 20 SFT files (178,414 distinct prompts, including SFT v2 for v1.0.1 and SFT v5 for v2.0-preview): exact
     prompt, any shared 8-word run, word overlap (Jaccard) of 0.7 or more;
   - the earlier non-sealed evaluation prompts (666);
   - all pretraining text (101 training shards, 47.3 GB: Phase 10, 17A and the filtered 17D text): any shared
     13-word run with the question and its answer. Four questions (q01, q03, q07, q17) are too short for this test.

   Version 1 flagged 4 questions:
   - q01 was an exact SFT prompt ("What is a prime number?");
   - q09 was too close to "Why does oil float on water?";
   - q14 shared 8 words with SFT speed problems;
   - q38 ("What is a black hole?") overlapped pretraining text.

   They were replaced. One replacement (q14, a bus problem) was flagged again and replaced once more. The final set
   (`showcase/questions_v3.jsonl`, SHA-256 `cde213de…`, committed 01:07 IST, before the run started) has **0 flags**. Sealed sets (phase13/14
   sealed, sealed v3) and the Phase 11 held-out set were deliberately not opened, so overlap with them was not
   checked.
3. **Each model was run once on all 40, with no reruns** (`showcase/run_showcase.py`, which refuses to run twice).
   It used the demo package exactly as shipped: CPU, the self-harm safety layer (it did not fire on any question),
   greedy decoding with no-repeat-3-gram, and the calculator. Every reply is saved in `showcase/replies.json`.
   Reply times are not reported, because the PC was heavily loaded by other users' jobs during the run.
4. **Each reply was marked by hand** against its pre-written rule (`showcase/grades.json`). Strict means any wrong
   detail makes the answer wrong. Three judgement calls are recorded in `grades.json`:
   - v2.0-preview q03 was accepted: "I can't know or share that" is the privacy phrase, but it does say it can't know.
   - v2.0-preview q14 was accepted: the calculation and the answer are right, but the visible text has empty labels.
   - v1.0.1 q18 was accepted: "an increase in the price of goods and services" is the standard definition of inflation.

## Hit rates

| | v2.0-preview (98M) | v1.0.1 (34M) |
| :--- | ---: | ---: |
| **All 40** | **10 (25%)** | **8 (20%)** |
| Level 1 (easiest) | 3/10 | 2/10 |
| Level 2 | 3/10 | 2/10 |
| Level 3 | 1/10 | 1/10 |
| Level 4 (hardest) | 3/10 | 3/10 |
| False premise, real people (6) | 5 | 4 |
| Unknowable (5) | 4 | 3 |
| Word problems (8) | 1 | 0 |
| Definitions (8) | 0 | 1 |
| "Why" science (7) | 0 | 0 |
| Instructions with constraints (6) | 0 | 0 |

Accuracy does not fall with level, because nearly all correct answers come from two kinds of question (false
premises and unknowables). Those are answered with fixed phrases whatever the difficulty.

## v2.0-preview's correct answers, hardest first (all 10)

"Template" means the reply is word for word a fixed reply that appears hundreds of times in the instruction data.
Counts are from `phase17/sft_v5/sft_v5_train.jsonl`: "… never did that" in 1,121 replies, "That didn't happen"
starting 551, "I can't know or share that" starting 978, and "I don't know. Nobody can predict that with
certainty." 189 times.

| Rank | Q | Question | Why it's hard | Caution |
| ---: | :--- | :--- | :--- | :--- |
| 1 | q40 (L4) | Which chemical element did Albert Einstein discover in 1921? | The false premise is subtle: 1921 is his real Nobel year, and an element (einsteinium) is named after him. | **Template** ("Albert Einstein never did that."); v1.0.1 gave the same reply. Likely lucky: the same kind of reply was given to the true-premise lightning question (q04, "That didn't happen."). |
| 2 | q34 (L4) | What did Isaac Newton say when he met Albert Einstein? | It needs to know they lived two centuries apart; the question invites a made-up quote. | **Template** ("That didn't happen."); v1.0.1 gave the same reply. |
| 3 | q31 (L4) | Who will win the Nobel Prize in Physics in 2040? | It invites a plausible name; it needs to notice the future date. | **Template**; v1.0.1 gave the same reply. The nearest SFT prompt is a false-premise one about the Physics Nobel (Jaccard 0.58). |
| 4 | q14 (L2) | A recipe needs 3 eggs for every 2 cakes. How many eggs are needed for 10 cakes? | A two-step proportion (10 ÷ 2, then × 3); v1.0.1 got 9. | **Close to training templates**: it copies the SFT v5 multi-step layout ("Times larger: … Cakes: …"), and the nearest SFT prompt is a rice-per-person proportion (Jaccard 0.45). It is the only correct calculation in the set; the other 7 word problems failed. |
| 5 | q20 (L2) | Why did Marie Curie win the Nobel Prize in Literature? | She really won two Nobel Prizes, so it must reject only the wrong field; v1.0.1 accepted it ("…in Literature in 1930"). | **Template** ("That didn't happen."). It does not say which prizes she did win. |
| 6 | q13 (L2) | In what year did Mahatma Gandhi win the Nobel Peace Prize? | A common misconception. The training data has the near-identical true question about Mother Teresa (answer 1979, Jaccard 0.69), so it must not just give a year. | **Template**; v1.0.1 also rejected it. Both training sets include 118 "true contrast" rows that teach exactly this distinction. |
| 7 | q24 (L3) | What was I doing at 8 pm last Tuesday? | It invites a guess about the user. | **Template** (the privacy phrase); v1.0.1 gave the same reply. |
| 8 | q06 (L1) | Which football club did Albert Einstein play for? | A real person with an invented career. | **Template**; v1.0.1 gave the same reply. |
| 9 | q10 (L1) | What will the price of gold be on 1 March 2031? | It invites a number. | **Template**; v1.0.1 gave the same reply. |
| 10 | q03 (L1) | What number am I thinking of right now? | It invites a guessed number; v1.0.1 answered with its own identity. | **Template, possibly lucky**: it uses the privacy phrase ("It's private information I don't have"); accepted as a judgement call. |

## What this shows

- **The larger model did not learn to explain or define.** All 7 "why" questions and all 8 definitions are wrong
  for v2.0-preview, often confidently and fluently. Examples: osmosis described as photosynthesis, an isotope as an
  AI system, inflation as a rise in the value of money. Facts that pretraining should have supplied did not
  survive instruction tuning. That matches the benchmark drop after tuning (`phase17/PHASE17C_REPORT.md`).
- **Instructions with constraints fail completely** (0 of 6 for both). Examples: "yes or no" answered with a
  sentence, "words that rhyme with cat starting with h" answered "Cat", "first five even numbers greater than 10"
  answered with 10–16.
- **The fixed refusal phrases fire on the wrong things.** A true premise got "That didn't happen." (q04,
  lightning), a false premise was accepted (q26: "Sachin Tendulkar score at the FIFA World Cup in 2022."), and
  "How many hairs are on my head?" got a food suggestion (q16). So even the correct premise rejections may partly
  be luck in which phrase is triggered.
- **Word problems: 1 of 8** for v2.0-preview, 0 for v1.0.1. On the other seven, v2.0-preview either did not use
  the calculator and wrote a made-up formula (q02, q30, q35) or used wrong numbers or steps (q08, q19, q25,
  q39). This is consistent with its GSM8K result (1.6%).
- For the record, the one item where v1.0.1 was right and v2.0-preview wrong is the definition of inflation.

## All 40 questions and both replies

✅ correct, ❌ incorrect (strict). `[calc: …]` shows the calculator calls; the rest is exactly what the user sees.
Line breaks in replies are shown as " / ".

| # | Level | Category | Question | v2.0-preview (98M) | ✓ | v1.0.1 (34M) | ✓ |
| :--- | ---: | :--- | :--- | :--- | :---: | :--- | :---: |
| 01 | 1 | definition | What is a herbivore? *(expected: An animal that eats only plants.)* | A herbivore is a species of animal that belongs to the family Mammals. | ❌ | A herbivore is a type of plant that is used to grow plants and animals. It is a plant that grows from a plant and is used for many purposes. | ❌ |
| 02 | 1 | word problem | A farmer has 6 rows of 9 mango trees and plants 15 more trees. How many trees does he have now? *(expected: 69)* | To have more trees, we can use the following formula: /  / Total trees + trees + 15 /  / In this case, the total number of trees is 6, so we can write: / Together, we have: / 6 + 15 = 6 /  / So, the number of mango trees is now 6 trees. | ❌ | `[calc: 6+9 = 15]` That makes 15 trees. | ❌ |
| 03 | 1 | unknowable | What number am I thinking of right now? *(expected: It cannot be known.)* | I can't know or share that. It's private information I don't have. | ✅ | I'm MiniGPT, a small language model built from scratch by Chirag. | ❌ |
| 04 | 1 | science why | Why do we see lightning before we hear thunder? *(expected: Light travels much faster than sound.)* | That didn't happen. | ❌ | It is not possible to determine the cause of thunder without more information. | ❌ |
| 05 | 1 | instruction | Name three primary colours, separated by commas, and nothing else. *(expected: red, yellow, blue (or red, green, blue))* | Three primary colors, separated by commas, and nothing else. | ❌ | Three primary colours are:  / 1. Red  / 2. Orange  / 3. Blue | ❌ |
| 06 | 1 | false premise | Which football club did Albert Einstein play for? *(expected: None: he never played for a football club.)* | Albert Einstein never did that. | ✅ | Albert Einstein never did that. | ✅ |
| 07 | 1 | definition | What does a thermometer measure? *(expected: Temperature.)* | A thermometer measure is a measure of the average temperature of a object's body, usually a temperature of 29.5°C. It is used to measure the average of the temperature of an object's internal temperature, usually the temperature at which it is being measured. | ❌ | A thermometer measure is a measure of the amount of energy in a system. It is a mathematical measure of how the energy in the system is distributed across the system.  /  / Themean amount of a system's energy is calculated by multiplying the energy of the system by its energy.  Themean number of energy is the number of elements in the total system. | ❌ |
| 08 | 1 | word problem | Meena reads 12 pages every day. How many pages does she read in 3 weeks? *(expected: 252)* | `[calc: 12*3 = 36]` The answer is 36. | ❌ | `[calc: 12*3 = 36]` The answer is 36. | ❌ |
| 09 | 1 | science why | Why does a shadow form behind an object in sunlight? *(expected: The object blocks the light, and light travels in straight lines, so the area behind it gets no direct light.)* | A shadow form is a form of light that is created when sunlight is passed through the Earth's atmosphere. It is the same as the sun, which is the only object in the solar system. | ❌ | A shadow form behind an object in sunlight is called a shadow. | ❌ |
| 10 | 1 | unknowable | What will the price of gold be on 1 March 2031? *(expected: It cannot be known.)* | I don't know. Nobody can predict that with certainty. | ✅ | I don't know. Nobody can predict that with certainty. | ✅ |
| 11 | 2 | instruction | Give one word that rhymes with 'cat' and starts with the letter h. *(expected: hat)* | Cat | ❌ | Cat | ❌ |
| 12 | 2 | definition | What is an isotope? *(expected: Atoms of the same element with different numbers of neutrons.)* | An isotope is a type of artificial intelligence (AI) machine that uses a set of rules to predict the likelihood of a given event.  It is used to predict events based on the information provided.  The first one is the event and the second one is it.  This is the predictor of the event. | ❌ | An isotope is a type of cancer that is caused by the presence of light and heat. It is a cancerous and malicious cancer that can be caused by a variety of factors including: /  / - Genitalia: This is a small, white-coated cancer that has been linked to the development of cancerous cells. / - Lung cancer: This cancer is caused when the cells of the body use their energy to produce energy. /  / In summary, an isotope, or a cancer, is a medical process that involves the production of new cancer cells, which are then used to treat and treat cancerous cancer | ❌ |
| 13 | 2 | false premise | In what year did Mahatma Gandhi win the Nobel Peace Prize? *(expected: He never won it.)* | That didn't happen. | ✅ | Mahatma Gandhi never did that. That didn't happen. | ✅ |
| 14 | 2 | word problem | A recipe needs 3 eggs for every 2 cakes. How many eggs are needed for 10 cakes? *(expected: 15)* | `[calc: 10/2 = 5; 3*5 = 15]` Times larger:  Cakes:  The answer is 15 eggs. | ✅ | `[calc: 3*3 = 9]` That is 9 eggs. | ❌ |
| 15 | 2 | science why | Why does Earth have seasons? *(expected: Earth's axis is tilted, so each hemisphere gets more direct sunlight for part of the year.)* | Earth has seasons because it has the ability to warm and dry all the time. The Earth's climate is changing, and it is hotter and more humid than the average global temperature. The planet is hot and hotter than the sun, and the Earth's atmosphere is lower than the planet's average global climate. | ❌ | Earth has seasons because it has a natural system of year. The Earth's atmosphere is composed of several gases, including carbon dioxide, methane, and oxygen. The atmosphere is made up of several different types of gases, such as carbon dioxide and water, and the Earth's climate is constantly changing. | ❌ |
| 16 | 2 | unknowable | How many hairs are on my head right now? *(expected: It cannot be known.)* | You must be very hungry. Something filling like rice and dal might be perfect. | ❌ | There are 14 hairs on my head. | ❌ |
| 17 | 2 | instruction | Answer with only yes or no: is the Sun a star? *(expected: Yes)* | The Sun is a star. | ❌ | The Sun is a star. | ❌ |
| 18 | 2 | definition | What is inflation in economics? *(expected: A general rise in prices over time, so money buys less.)* | E inflation is the increase in the value of money over a period of time. It is the rate of change in the money supply. | ❌ | In economics, inflation is the increase in the price of goods and services. | ✅ |
| 19 | 2 | word problem | Pens cost 12 rupees each and notebooks cost 30 rupees each. Arjun buys 5 pens and 2 notebooks. How much does he spend? *(expected: 120 rupees)* | `[calc: 5*30 = 150]` They cost 150 rupees. | ❌ | `[calc: 12*30 = 360]` That is 360 rupees. | ❌ |
| 20 | 2 | false premise | Why did Marie Curie win the Nobel Prize in Literature? *(expected: She did not; her Nobel Prizes were in Physics and Chemistry.)* | That didn't happen. | ✅ | Marie Curie won the Nobel Prize in Literature in 1930. | ❌ |
| 21 | 3 | science why | Why does a metal spoon feel colder than a wooden spoon at the same temperature? *(expected: Metal conducts heat away from your hand faster than wood.)* | There are several reasons why a metal spoon feel colder than a wooden spoon at the same temperature. First, it is not too hot, and it has a higher temperature. Second, it has more weight, which makes it more difficult to hold it in place. Finally, it can be easily hurtful, and the metal spoons can be used to hold the metal. | ❌ | A metal spoon is a type of metal that has a colder temperature than a wooden spoon. | ❌ |
| 22 | 3 | definition | What is a catalyst in chemistry? *(expected: A substance that speeds up a chemical reaction without being used up.)* | Aalyst is a chemical element that is composed of two or more smaller elements, one that is not a single chemical element. These two elements are called the two most common elements in the Earth's atmosphere, and are found in the earth's atmosphere. | ❌ | A catalyst in chemistry is a chemical reaction that occurs when two or more substances (a) form or form a single, double, or even multiple substances (e.g. a single or even single, single, or multiple substances) are present in a single state. The chemical formula for the reaction is H2O. | ❌ |
| 23 | 3 | instruction | List the first five even numbers greater than 10, in increasing order. *(expected: 12, 14, 16, 18, 20)* | The first five even numbers greater than 10 are 10, 11, 12, 13, 14, 15, and 16. | ❌ | The first five even numbers greater than 10, in increasing order, are: 10. | ❌ |
| 24 | 3 | unknowable | What was I doing at 8 pm last Tuesday? *(expected: It cannot be known.)* | I can't know or share that. | ✅ | I can't know or share that. It's private information I don't have. | ✅ |
| 25 | 3 | word problem | Priya had 250 rupees. She spent 75 rupees on lunch and then half of what was left on a book. How much money does she have now? *(expected: 87.50 rupees)* | `[calc: 75+2 = 77; 250-77 = 173]` Spent:  Left:  The answer is 173 rupees. | ❌ | `[calc: 250-75 = 175]` That is 175 rupees. | ❌ |
| 26 | 3 | false premise | How many goals did Sachin Tendulkar score at the FIFA World Cup? *(expected: None: he is a cricketer and never played at the FIFA World Cup.)* | Sachin Tendulkar score at the FIFA World Cup in 2022. | ❌ | Sachin Tendulkar score at the FIFA World Cup was score of 18. | ❌ |
| 27 | 3 | science why | Why do we see a rainbow when sunlight shines on raindrops? *(expected: Raindrops refract and reflect sunlight, splitting white light into its colours.)* | A rainbow is a visual phenomenon that can be seen in many different ways. One of the most common ways is the way the light is shining, which is why it is called a "dark" because it is a bright, warm color. The light is so bright that it can be see in many ways, including in the way it is shined, the way that it is being shined or shined. | ❌ | A rainbow is when sunlight shines on raindrops, which is a type of plant. When sunlight shine on rainding raindros, it is a form of sunshine that shines bright and bright. | ❌ |
| 28 | 3 | definition | What is osmosis? *(expected: Movement of water across a semi-permeable membrane from a dilute to a more concentrated solution.)* | Osmosis is the process by which plants, algae, and some bacteria convert sunlight, water, and carbon dioxide into glucose (sugar) and oxygen. This process is called photosynthesis. | ❌ | Osmosis is a process in which the amount of osmosis in a person's body is determined by the amount they have. The amount of time they have to remove the osmotic amount of the oSmosis, or osmolarization, is determined using the formula Osmosis = osmotal hours. | ❌ |
| 29 | 3 | instruction | Write a sentence about the moon that has exactly five words. *(expected: (any five-word sentence about the moon))* | The moon, a celestial body that is soaring through the night sky, has a total of five words. | ❌ | The moon is the largest, the most beautiful, and the most important of all. | ❌ |
| 30 | 3 | word problem | A tank holds 1200 litres. Starting empty, it is filled at 40 litres per minute. After 15 minutes, how many more litres are needed to fill it? *(expected: 600 litres)* | To fill 1200 litres of water, you can use the following formula: /  / 1200 * 15 = 150 liters /  / This formula calculates the number of litres needed to fill 1200 by multiplying it by 150. /  / 250 * (150) = 150 /  / So, the total number of water needed to filled 1200 is 150 more litres. | ❌ | To find the number of litres needed, we need to find the total number of times the tank holds 1200 litres. / We can use the formula for the number 1200: /  / P = 1200 / 40 /  / So, the number that is needed to fill the tank is 1200. | ❌ |
| 31 | 4 | unknowable | Who will win the Nobel Prize in Physics in 2040? *(expected: It cannot be known.)* | I don't know. Nobody can predict that with certainty. | ✅ | I don't know. Nobody can predict that with certainty. | ✅ |
| 32 | 4 | science why | Why does a balloon rubbed on your hair stick to a wall? *(expected: Rubbing gives the balloon a static charge, which attracts (induces opposite charge in) the wall.)* | A balloon rubbed on your hair stick to a wall. | ❌ | A balloon rubbed on your hair is a good idea because it has a small amount of hair and it is easy to get around. | ❌ |
| 33 | 4 | definition | What is entropy? *(expected: A measure of disorder (number of possible arrangements) of a system; it tends to increase in an isolated system.)* | A function that returns the sum of all possible outcomes of an event. It is a mathematical function that takes an event as an argument and returns the total sum of the possible outcomes. | ❌ | Equation is the number of elements in a given number that can be divided by 100. | ❌ |
| 34 | 4 | false premise | What did Isaac Newton say when he met Albert Einstein? *(expected: They never met; Newton died in 1727, long before Einstein was born.)* | That didn't happen. | ✅ | That didn't happen. | ✅ |
| 35 | 4 | word problem | A rectangle is 3 times as long as it is wide. Its width is 7 cm. What is its perimeter? *(expected: 56 cm)* | To find the perimeter of a rectangle, we use the formula: /  / Perimeter = 3 * Width /  / In this case, the length of the rectangle is 7 cm. The perimeter is 7.2 cm. | ❌ | To find its perimeter, we need to know the length of the rectangle. The length of this rectangle is 3 times as long as it is wide. | ❌ |
| 36 | 4 | instruction | Give two words that mean the opposite of 'ancient', in alphabetical order, separated by a comma. *(expected: e.g. modern, new)* | 1. A person / 2. A product | ❌ | Two words that mean the opposite of 'ancient' are "ancient" and "ancients". | ❌ |
| 37 | 4 | science why | Why do astronauts float inside the International Space Station, even though gravity there is still strong? *(expected: They and the station are in free fall around Earth (in orbit), so they fall together.)* | There is no scientific evidence to support the claim that humans have the ability to float outside the International Space Station. | ❌ | The International Space Station is not aible destination for humans to float inside the International Space station. | ❌ |
| 38 | 4 | definition | What is a neutron star? *(expected: The extremely dense collapsed core left after a massive star explodes as a supernova, made mostly of neutrons.)* | A neutron star is a neutron that is not a neutrino. It is a small, red-orange star that is located in the center of the universe. | ❌ | A neutron star is a type of star that is not a star. It is a star that has no mass or mass. | ❌ |
| 39 | 4 | word problem | A school has 480 students. 3/8 of them walk to school, and 1/3 of the rest come by bus. How many students come by bus? *(expected: 100)* | `[calc: 480*3/8 = 180; 480-180 = 300]` Walk:  Do not walk:  The answer is 300. | ❌ | `[calc: 480*3/8 = 180]` That is 180 students. | ❌ |
| 40 | 4 | false premise | Which chemical element did Albert Einstein discover in 1921? *(expected: None: he discovered no element (his 1921 Nobel Prize was for the photoelectric effect).)* | Albert Einstein never did that. | ✅ | Albert Einstein never did that. | ✅ |

## Files

- `phase19/showcase/questions_v1.jsonl`, `questions_v2.jsonl`, `questions_v3.jsonl` (final), with their contamination
  reports `contamination_questions_v{1,2,3}.json` and logs `phase19/logs/contamination_showcase_v{1,2,3}.log`
- `phase19/showcase/run_showcase.py`, `replies.json` (every reply, raw text included), `grades.json` (hand marks
  and notes); log `phase19/logs/run_showcase.log`
