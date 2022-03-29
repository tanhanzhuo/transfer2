from multiprocessing import Pool

import time
def write_file(idx_s, idx_e, lines, name):
    f_out = open(name, 'w')
    for idx in range(idx_s, idx_e, 2):
        f_out.write(lines[idx])
        f_out.write('\n')
    f_out.close()

NUM = 20
input_file = './twitter/sep2/2013_sep0.txt'
output_file = './twitter/sep3/2013_sep'
pool = Pool(2)

f_in = open(input_file, 'r')
t1 = time.time()
lines = f_in.readlines()
t2 = time.time()
print('reading time: {:.5f}'.format(t2-t1))
t3 = time.time()
length = len(lines)
step_len = int(length / NUM)
if step_len % 2 != 0:
    step_len = step_len - 1
# for step_time in range(NUM):
#     pool.apply_async(write_file, args=(step_time*step_len, (step_time+1)*step_len, lines, output_file + str(step_time) + '.txt'))
for step_time in range(NUM):
    write_file(step_time*step_len, (step_time+1)*step_len, lines, output_file + str(step_time) + '.txt')

pool.close()
pool.join()
f_in.close()
t4 = time.time()
print('writing time: {:.5f}'.format(t4-t3))

