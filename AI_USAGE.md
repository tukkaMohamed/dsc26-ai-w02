# AI usage

## Tools I used, and for which functions

I used ChatGPT to understand the task requirements, review edge cases, and help implement and check the logic in `rawaj/audit.py` and `rawaj/metrics.py`. I used the public pytest suite to check the implementation.

## One place where the AI was wrong, incomplete, or not what the spec said

An initial test run failed because some metric functions still contained `NotImplementedError`. I compared the failures with the required function behavior, completed the implementations, and reran the tests. The public test suite then passed: 5 tests passed.

## The function I wrote without AI, and why

I wrote `precision_at_k` myself to practise stable ranking, handling tied scores, and calculating precision among the highest-ranked cases. This helped me understand how to prioritize a limited number of transfers for manual review.