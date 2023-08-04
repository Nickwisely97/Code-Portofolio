seed = 3

# easy message
# sending_msg = "Bersatu kita teguh, bercerai kita runtuh."
# medium message
sending_msg = "Indonesia tanah air beta. Pusaka Abadi nan Jaya. Indonesia sejak dulu kala, tetap dipuja-puja bangsa."

msg1 = sending_msg.split(" ")

encr_msg = [None]*seed
for i in range(seed):
    encr_msg[i] = " ".join(msg1[i::seed])
print(encr_msg)

encr_result = " ".join(encr_msg)
print(f'''
    This is the encrypted message:
    {encr_result}
'''
)

msg2 = encr_result.split(" ")
for i in range(seed):
    if len(msg2)%seed==i:
        num = (len(msg2) + (seed-i)) // seed

decr_msg = [None]*num
for i in range(num):
    decr_msg[i] = " ".join(msg2[i::num])

print(f'''
    This is the decoded message:
    {" ".join(decr_msg)}
'''
)
