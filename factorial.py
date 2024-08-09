# write a py program to write a factorial
# 3 rows 9 column
# write a program to generate aa tambola ticket
def factorial(num):
    if num == 0 or num==1:
        return 1
    return num*factorial((num-1))

print(factorial(0))

#