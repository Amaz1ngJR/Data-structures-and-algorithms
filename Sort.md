
快速排序和归并排序都是基于分治思想的
![image](https://github.com/Amaz1ngJR/Re/assets/83129567/89d82c34-a238-4475-b9bc-ea22c4040c76)

#  *插入排序（空间复杂度为O(1)，不适用链表）

## 直接插入排序O(n^2)（gap为1的希尔排序）
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/5424a3b5-c000-428e-9602-b035ed77f9cd)

```c++
//直接插入排序
//性能分析 稳定
//时间复杂度：平均O(n^2) 最好O(n) 最坏O(n^2)
//空间复杂度：O(1)
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

## 折半插入排序O(n^2)

将直接插入排序中，遍历有序序列改为二分查找。

```c++
//折半插入排序
//性能分析 稳定
//时间复杂度：平均O(n^2) 最好O(n) 最坏O(n^2)
//空间复杂度：O(1)
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
## 希尔排序/缩小增量排序
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/9bf01f8f-c80a-4ae6-aab2-07c78d9b8d39)

```c++
//希尔排序
//性能分析 不稳定
//时间复杂度：平均O(n^1.3-1.5) 最好O(n) 最坏O(n^2)
//空间复杂度：O(1)
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

# *交换排序

## 快速排序O（nlogn）（重要）

```c++
//快速排序
//性能分析 不稳定
//时间复杂度：平均O(nlogn) 最好O(nlogn) 最坏O(n^2)
//空间复杂度：O(nlogn)
template<typename T>
void QuickSort(vector<T>& nums, int low, int high) {
    //划分
    std::function<int(vector<T>&, int, int)>Parttion =
        [&](vector<T>& nums, int low, int high)->int {
        int pivot = nums[low];//每次划分确定基准pivot:nums[low]的位置
        while (low < high) {
            while (low < high && nums[high] >= pivot) --high;
            nums[low] = nums[high];//nums[high]<pivot
            while (low < high && nums[low] <= pivot) ++low;
            nums[high] = nums[low];//nums[low]>pivot
        }
        nums[low] = pivot;//基准的位置为当前的low
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

## 冒泡排序O(n^2)

![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/d7b7ba2c-f66c-4c38-8f12-9cde8b7a7488)


```C++
//冒泡排序
//性能分析 稳定
//时间复杂度：平均O(n^2) 最好O(n) 最坏O(n^2)
//空间复杂度：O(1)
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

# *选择排序

## 简单选择排序O(n^2)

![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/fee239fd-9441-4465-9fc6-73802c73819f)

```c++
//简单选择排序
//性能分析 不稳定
//时间复杂度：平均O(n^2) 最好O(n^2) 最坏O(n^2)
//空间复杂度：O(1)
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

## 堆排序O(nlogn)

```C++
//堆排序
//性能分析 不稳定
//时间复杂度：平均O(nlogn) 最好O(nlogn) 最坏O(nlogn)
//空间复杂度：O(1)
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

# *归并排序

## 二路归并

```c++
//二路归并排序
//性能分析 稳定
//时间复杂度：平均O(nlogn) 
//空间复杂度：O(n)
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

## 多路归并（外部排序）

# * 基数排序

# * 桶排序
