# Algorithm

## *排序算法

快速排序和归并排序都是基于分治思想的

###  **插入排序（空间复杂度为O(1)，不适用链表）

#### ***直接插入排序O(n^2)（gap为1的希尔排序）

```c++
//ֱ直接插入排序
template<typename T>
void DirenctlyInsertSort(vector<T>& nums) {
	for (int i = 1; i < nums.size(); i++) {
		int temp = nums[i];
		int j;
		for (j = i - 1; j >= 0 && nums[j] > temp; j--) {
			nums[j + 1] = nums[j];
		}
		nums[j + 1] = temp;
	}
}
```



#### ***折半插入排序O(n^2)

将直接插入排序中，遍历有序序列改为二分查找。

```c++
//折半插入排序
template<typename T>
void BinarySearchSort(vector<T>& nums) {
	for (int i = 1; i < nums.size(); i++) {
		if (nums[i] < nums[i - 1]) {
			int temp = nums[i];
			int left = 0;
			int right = i - 1;
			while (left <= right) {
				int mid = (right - left) / 2 + left; //防溢出
				if (nums[mid] > temp)
					right = mid - 1;
				else left = mid + 1;
			}
			for (int j = i - 1; j >= left; j--) {
				nums[j + 1] = nums[j];
			}
			nums[left] = temp;
		}
	}
}
```



#### ***希尔排序/缩小增量排序

```c++
	//希尔排序
	template<typename T>
	void ShellSort(vector<T>& nums) {
		//希尔排序
		std::function<void(vector<T>&, int)> Shell =
			[&](vector<T>& nums, int gap) {
			for (int i = gap; i < nums.size(); i++) {
				int temp = nums[i];
				int j;
				for (j = i - gap; j >= 0 && nums[j] > temp; j -= gap) {
					nums[j + gap] = nums[j];
				}
				nums[j + gap] = temp;
			}
		};
		int gap = nums.size();
		while (gap > 1) {
			gap /= 2;
			Shell(nums, gap);
		}
	}
```



### **交换排序

#### ***快速排序O（nlogn）（重要）

```c++
	//快速排序
	template<typename T>
	void QuickSort(vector<T>& nums, int low, int high) {
		//划分
		std::function<int(vector<T>&, int, int)>Parttion =
			[&](vector<T>& nums, int low, int high)->int {
			int pivot = nums[low];
			while (low < high) {
				while (low < high && nums[high] >= pivot) --high;
				nums[low] = nums[high];
				while (low < high && nums[low] <= pivot) ++low;
				nums[high] = nums[low];
			}
			nums[low] = pivot;
			return low;
		};
		//排序
		if (low < high) {
			int pivotpos = Parttion(nums, low, high);
			QuickSort(nums, low, pivotpos - 1);
			QuickSort(nums, pivotpos + 1, high);
		}
	}
```



#### ***冒泡排序O(n^2)

```C++
	//冒泡排序
	template<typename T>
	void BubbleSort(vector<T>& nums) {
		for (int i = 0; i < nums.size(); i++) {
			for (int j = 0; j + 1 < nums.size() - i; j++) {
				if (nums[j] > nums[j + 1]) {
					swap(nums[j], nums[j + 1]);
				}
			}
		}
	}
```



### ** 选择排序

#### ***简单选择排序O(n^2)

```c++
	//简单选择排序
	template<typename T>
	void SelectSort(vector<T>& nums) {
		for (int i = 0; i < nums.size() - 1; i++) {
			for (int j = i + 1; j < nums.size(); j++) {
				if (nums[j] < nums[i]) {
					swap(nums[j], nums[i]);
				}
			}
		}
	}
```



#### ***堆排序O(nlogn)

```C++
//堆排序
template<typename T>
void HeapSort(vector<T>& nums) {
	//调整堆
	std::function <void(vector<T>&, int, int)> HeapAdjust =
		[&](vector<T>& nums, int start, int end) {
		for (int i = 2 * start + 1; i <= end; i = i * 2 + 1) {
			//大根堆
			if (i < end && nums[i] < nums[i + 1]) {//有左孩子，且左孩子小于右孩子
				i++;//用i表示孩子中最大的
			}
			if (nums[i] > nums[start]) {
				swap(nums[start], nums[i]);
				start = i;
			}
			else break;
		}
	};
	//建立大根堆，从后往前调整
	for (int i = (nums.size() - 2) / 2; i >= 0; i--)
		HeapAdjust(nums, i, nums.size() - 1);
	for (int i = 0; i < nums.size() - 1; i++) {
		swap(nums[0], nums[nums.size() - 1 - i]);
		HeapAdjust(nums, 0, nums.size() - 2 - i);
	}
}
```

### **归并排序

#### ***二路归并

```c++
template<typename T>
void MergeSort(vector<T>& a, int low, int high) {
	int i, j, k;
	vector<int> b;//辅助数组
	std::function<void(vector<T>&, int, int, int)>Merge =
		[&](vector<T>& a, int low, int mid, int high) {
		b.resize(a.size());
		for (k = low; k <= high; k++) b[k] = a[k];//将a中所有元素复制到b
		for (i = low, j = mid + 1, k = i; i <= mid && j <= high; k++) {
			if (b[i] <= b[j]) a[k] = b[i++]; //较小值复制到a
			else a[k] = b[j++];
		}
		while (i <= mid) a[k++] = b[i++];
		while (j <= high) a[k++] = b[j++];
	};
	if (low < high) {
		int mid = (low + high) / 2;    //从中间划分
		MergeSort(a, low, mid);        //对左半部分归并排序
		MergeSort(a, mid + 1, high);   //对右半部分归并排序
		Merge(a, low, mid, high);      //归并
	}
}
```

#### ***多路归并（外部排序）

### ** 基数排序



### ** 桶排序



##  *查找算法

### **二分/折半查找O（logn）

```c++
//二分查找 有序数组
template<typename T>
int BinarySearch(vector<T>& nums, T target) {
	//闭区间写法[left,right]
	int left = 0;
	int right = nums.size() - 1;
	int mid = (left + right) / 2;
	if (nums[0] >= target) return 0;
	if (nums[nums.size() - 1] < target) return nums.size();
	while (left <= right) {
		if (nums[mid] == target) return mid;//找到返回mid
		if (nums[mid] > target) right = mid - 1;
		if (nums[mid] < target) left = mid + 1;
		mid = (left + right) / 2;
	}
	return left;//找不到就返回target应该插入的位置

	//开区间写法(low,high)
	//int low = -1;
	//int high = nums.size();
	//int mid;
	//while (low + 1 < high) {
	//	mid = low + ((high - low) / 2);
	//	if (nums[mid] == target)return mid;
	//	else if (nums[mid] > target)
	//		high = mid;
	//	else
	//		low = mid;
	//}
	//return high;//找不到就返回target应该插入的位置
}
```



## *前后缀

### **前缀和 与 差分



### **前缀和最值



### **分解前后缀



## *双指针

### **相向双指针



### **同向双指针 滑动窗口



### **快慢指针



### **前后指针



## *递归 二叉树



## *回溯

### **子集型



### **组合型



### **排列型



## *动态规划



### **打家劫舍



### **0/1背包



### **完全背包



### **子序列(连续)



## *贪心

