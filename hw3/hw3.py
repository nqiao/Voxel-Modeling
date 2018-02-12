import numpy as np
import matplotlib.pyplot as plt

# Problem 1
# 1.a
def analysis(ori_arr):
    """Analyze the input array, return the first array and the second array."""
    even_res = []
    odd_res = []
    for i in range(len(ori_arr)):
        if i % 2 == 0:
            even_res.append((ori_arr[i+1] + ori_arr[i]) / 2)
        else:
            odd_res.append((ori_arr[i] - ori_arr[i-1]) / 2)
    return np.array(even_res), np.array(odd_res)

def plot_analysis(ori_arr, first_arr, second_arr, fig_name="no_name"):
    n_arr = len(ori_arr)
    plt.figure()
    plt.plot(ori_arr, 'o', label='original array')
    plt.plot([i for i in range(0,n_arr-1,2)], first_arr, 'o', label='first array')
    plt.plot([i for i in range(1,n_arr,2)], second_arr, 'o', label='second array')
    plt.legend()
    plt.savefig(fig_name + ".jpg")

def synthesis(first_arr, second_arr):
    """Do synthesis based on the two given arrays, return the reconstructed array."""
    syn_arr = np.stack((first_arr, second_arr))
    syn_arr = syn_arr.T
    syn_arr = syn_arr.flatten()
    n_arr = len(syn_arr)
    syn_res = np.empty(n_arr)
    for i in range(n_arr):
        if i % 2 == 0:
            syn_res[i] = syn_arr[i] - syn_arr[i+1]
        else:
            syn_res[i] = syn_arr[i-1] + syn_arr[i]    
    return syn_res

print("### Problem 1 ###")

# 1.b
print("--- 1.b ---")
test_1 = np.ones(16)
test_2 = np.linspace(0, 1, 16)
test_3 = np.linspace(0, 1, 16) ** 2
test_4 = np.random.rand(16)
print("test_1: ", test_1)
print()
print("test_2: ", test_2)
print()
print("test_3: ", test_3)
print()
print("test_4: ", test_4)
print()

# 1.c
print("--- 1.c ---")
test_1_even, test_1_odd = analysis(test_1)
plot_analysis(test_1, test_1_even, test_1_odd, fig_name='1.c.1')
print("test_1 first array: ", test_1_even)
print("test_1 second array: ", test_1_odd)
print("Plot see figure test_1")
print()

test_2_even, test_2_odd = analysis(test_2)
plot_analysis(test_2, test_2_even, test_2_odd, fig_name='1.c.2')
print("test_2 first array: ", test_2_even)
print("test_2 second array: ", test_2_odd)
print("Plot see figure test_2")
print()

test_3_even, test_3_odd = analysis(test_3)
plot_analysis(test_3, test_3_even, test_3_odd, fig_name='1.c.3')
print("test_3 first array: ", test_3_even)
print("test_3 second array: ", test_3_odd)
print("Plot see figure test_3")
print()

test_4_even, test_4_odd = analysis(test_4)
plot_analysis(test_4, test_4_even, test_4_odd, fig_name='1.c.4')
print("test_4 first array: ", test_4_even)
print("test_4 second array: ", test_4_odd)
print("Plot see figure test_4")
print()

# 1.d
print("--- 1.d ---")
syn_1 = synthesis(test_1_even, test_1_odd)
print("synthesis result 1: ", syn_1)
print("difference with the original array: ", syn_1 - test_1)
print()

syn_2 = synthesis(test_2_even, test_2_odd)
print("synthesis result 2: ", syn_2)
print("difference with the original array: ", syn_2 - test_2)
print()

syn_3 = synthesis(test_3_even, test_3_odd)
print("synthesis result 3: ", syn_3)
print("difference with the original array: ", syn_3 - test_3)
print()

syn_4 = synthesis(test_4_even, test_4_odd)
print("synthesis result 4: ", syn_4)
print("difference with the original array: ", syn_4 - test_4)
print()

# 1.e
print("--- 1.e ---")
zero_array = np.zeros(8)
syn_1_0 = synthesis(test_1_even, zero_array)
print("synthesis result 1: ", syn_1_0)
print("difference with the original array: ", syn_1_0 - test_1)
print()

syn_2_0 = synthesis(test_2_even, zero_array)
print("synthesis result 2: ", syn_2_0)
print("difference with the original array: ", syn_2_0 - test_2)
print()

syn_3_0 = synthesis(test_3_even, zero_array)
print("synthesis result 3: ", syn_3_0)
print("difference with the original array: ", syn_3_0 - test_3)
print()

syn_4_0 = synthesis(test_4_even, zero_array)
print("synthesis result 4: ", syn_4_0)
print("difference with the original array: ", syn_4_0 - test_4)
print()
