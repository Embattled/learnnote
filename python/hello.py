# Assignment III Part B

# No. 1, No. 2, and No. 3
numbers=[]

while True:
  
  number=input("What is your favorite integer? ")
  if number=='quit':
    break
  number=int(number)
  numbers.append(number)
  remainder=number%3
  print(f"Do you know that if {number} is divided by 3, the remainder is {remainder}?")

  if remainder==0:
    print("Your number is divisible by 3!")
  else:
    print("Your number is not divisible by 3!")


# No. 4 and No. 5
print("Here are the numbers you've typed in.")
print(numbers)
