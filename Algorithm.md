# Algorithm
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/3f60b9b4-4d9a-4d77-8bca-82876d343086)

## 二分/折半查找

适用：排好序的数组
```c++
//二分查找 有序数组
template<typename T>
int BinarySearch(vector<T>& nums, T target) {
	//时间复杂度O(logn)
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

### *二分查找的变形
pre [852. 山脉数组的峰顶索引](https://leetcode.cn/problems/peak-index-in-a-mountain-array/)
```c++
int peakIndexInMountainArray(vector<int>& arr) {
	int left = 0, right = nums.size() - 1, ans = 0;
	while (left < right) {
		int mid = left + (right - left) / 2;
		if (nums[mid] > nums[mid + 1]) {
			ans = mid;
			right = mid - 1;
		}
		else left = mid;
	}
	return ans;
}
```
pre plus [845. 数组中的最长山脉](https://leetcode.cn/problems/longest-mountain-in-array/)
#### [162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/)

```c++
int findPeakElement(vector<int>& nums) {
    int low = -1, high = nums.size() - 1, mid; //开区间 (-1, n-1)
    while (low+1 < high) { // 开区间不为空
        mid = low + ((high - low) / 2);
        if (nums[mid] > nums[mid + 1]) high = mid;
        else low = mid ;
    }
    return high;
}
```

#### [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

```c++
int findMin(vector<int>& nums) {
    int low = -1;
    int high = nums.size()-1;
    int mid;
    while (low + 1 < high) {
        mid = low + ((high - low) / 2);
        if (nums[mid] < nums[nums.size()-1])
            high = mid;
        else
            low = mid;
    }
    return nums[high];

}
```

#### [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

可以使用153的方法先找到两个有序数组，再二分。仅一次二分的方法如下

二分位置nums[mid]右侧全是蓝色的三种情况：

1、如果nums[n-1]<target<=nums[mid] 同一左段 

2、nums[mid]<nums[n-1]<target    target左段、mid右段

3、target<=nums[mid]<nums[n-1]    同一右段   

```c++
int search(vector<int>& nums, int target) {
    int n = nums.size();
    if (nums[n - 1] == target)return n - 1;
    int low = -1;
    int high = nums.size() - 1;
    int mid;
    auto isblue = [&](int mid) {
        if (nums[mid] > target) {
            if (target > nums[n - 1]) return true;
            else if (nums[mid] < nums[n - 1]) return true;
        }
        else if (target > nums[n - 1] && nums[n - 1] > nums[mid])return true;
        return false;
    };
    while (low + 1 < high) {
        mid = low + ((high - low) / 2);
        if (nums[mid] == target)return mid;
        else if (isblue(mid))
            high = mid;
        else
            low = mid;
    }
    return -1;
}
```
#### [1901. 寻找峰值 II](https://leetcode.cn/problems/find-a-peak-element-ii/)
```c++
vector<int> findPeakGrid(vector<vector<int>>& mat) {
	int m = mat.size();
	int low = 0, high = m - 1, i, j;
	while (low <= high) {
		i = low + (high - low) / 2;
		j = distance(mat[i].begin(), max_element(mat[i].begin(), mat[i].end()));
		if (i - 1 >= 0 && mat[i][j] < mat[i - 1][j]) {//小于上方
			high = i - 1;
			continue;
		}
		if (i + 1 < m && mat[i][j] < mat[i + 1][j]) {//小于下方
			low = i + 1;
			continue;
		}
		return{ i,j };
	}
	return{ -1,-1 };
}
```
## 前后缀与差分

```
前缀和sum[i] = accumulate(arr[0],arr[i])
sum[L,R] = sum[R] - sum[L-1]
差分数组dif[i+1] = sum[i+1] - sum[i]
```
应用：对一个区间[L,R]所有元素加上值v 就转换成了只对差分数组的两个元素分别加、减一个v 然后进行一次前缀和
```
[L,R] + v == dif[L] + v , dif[R+1] – v ;  sumd[L,R];
例如：
arr:   1, 3, 7, 5, 2
dif:   1, 2, 4,-2,-3
sumd:  1, 3, 7, 5, 2  //还原 原数组
//在arr[1,3]区间元素+3
dif2:  1, 5, 4,-2,-6//仅对dif[1]+3 dif[4]-3
sumd2: 1, 6,10, 8, 2
```
二维前缀和

![bafcdc70864e80ef1a00fae5ad94f349](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/fbd57c75-c76a-4f8e-bff8-0fd898a80af9)

```c++
vector<vector<int>>sum(m + 1, vector<int>(n + 1, 0));//二维前缀和 下标从1开始
for (int i = 0; i < m; i++) {
	for (int j = 0; j < n; j++) {
		sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + grid[i][j];
	}
}
int s = sum[x][y] - sum[x][j - 1] - sum[i - 1][y] + sum[i - 1][j - 1]; //蓝色区域的和
```

二维差分

![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/8357a9c9-8c01-4f0c-b91b-468735aedf26)

```
dif[0][0] = matrix[0][0]
第一行:dif[0][j] = matrix[0][j] - matrix[0][j-1]
第一列:dif[i][0] = matrix[i][0] - matrix[i-1][0]
其他元素:dif[i][j] = matrix[i][j] - matrix[i-1][j] - matrix[i][j-1] + matrix[i-1][j-1]
例如：
1  2  4  3        1  1  2 -1
5  1  2  4    =>  4 -5 -1  3
6  3  5  9        1  1  1  2  
```
还原原矩阵
```
还原第一行元素：for(int j = 1; j < col; j++) dif[0][j] += dif[0][j-1]; matrix[0][j] = dif[0][j];
还原第一列元素：for(int i = 1; i < row; i++) dif[i][0] += dof[i-1][0]; matrix[i][0] = dif[i][0];
还原其他元素：matrix[i][j] = diff[i][j] + matrix[i-1][j] + matrix[i][j-1] - matrix[i-1][j-1]
```

应用：在(x1,y1)到(x2,y2)的区间加v等效于
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/7fc1590f-f97d-4b3b-8e61-9147b67cfaba)
```c++
dif[x1][y1] += v;  //图中第一行第二个全黄区域都+v
dif[x1][y2+1] -= v; //减去图中第一行第三个蓝色区域
dif[x2+1][y1] -= v;  //减去图中第二行第一个蓝色区域
dif[x2+1][y2+1] += v; //加上两个蓝色区域多减的绿色区域
```

### *差分
[1094. 拼车](https://leetcode.cn/problems/car-pooling/)
```c++
bool carPooling(vector<vector<int>>& trips, int capacity) {
	vector<int> dif(1001, 0);
	for (const auto& t : trips) {
		dif[t[1]] += t[0];
		dif[t[2]] -= t[0];
	}
	int sum = 0;
	for (const int& v : dif) {
		sum += v;
		if (sum > capacity)return false;
	}
	return true;
}
```
[2132. 用邮票贴满网格图](https://leetcode.cn/problems/stamping-the-grid/)
```c++
bool possibleToStamp(vector<vector<int>>& grid, int stampHeight, int stampWidth) {
	int m = grid.size(), n = grid[0].size();
	//1、计算二维前缀和
	vector<vector<int>>sum(m + 1, vector<int>(n + 1, 0));//二维前缀和 下标从1开始
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + grid[i][j];
		}
	}
	//2、计算二维差分
	vector<vector<int>>dif(m + 2, vector<int>(n + 2, 0));//二维差分数组 扩充第0行 第0列 
	for (int i = stampHeight; i <= m; i++) {
		for (int j = stampWidth; j <= n; j++) {
			//左上角(i0,j0) 右下角(i,j) 
			int i0 = i - stampHeight + 1;
			int j0 = j - stampWidth + 1;
			if (sum[i][j] - sum[i][j0 - 1] - sum[i0 - 1][j] + sum[i0 - 1][j0 - 1] == 0) {//区域内无被占格子
				//区域内+1表示贴了一张邮票
				dif[i0][j0]++;
				dif[i0][j + 1]--;
				dif[i + 1][j0]--;
				dif[i + 1][j + 1]++;
			}
		}
	}
	//3、还原矩阵(原地计算)
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			dif[i + 1][j + 1] += dif[i + 1][j] + dif[i][j + 1] - dif[i][j];
			if (!grid[i][j] && !dif[i + 1][j + 1])return false;//没被占据&&没有贴邮票
		}
	}
	return true;
}
```
### *前缀和最值

#### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

```c++
int maxProfit(vector<int>& prices) {
    vector<int>suf;
    int ans = 0;
    int n = prices.size();
    suf.resize(n);
    int  maxsuf= 0;
    for (int j = n - 1; j >= 0; j--) {
        suf[j] = maxsuf;
        maxsuf = max(maxsuf, prices[j]);
    }
    for (int i = 0; i < n; i++) {
        ans = max(ans, suf[i] - prices[i]);
    }
    return ans;
}
```

### *分解前后缀

#### [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)

```c++
vector<int> productExceptSelf(vector<int>& nums) {
    int n = nums.size();
    vector<int> ans(n, 1);
    int i, suf;
    suf = 1;
    for (i = 1; i < n; i++) {
        ans[i] = nums[i - 1] * ans[i - 1];//ans是前缀积
    }
    for (i = n - 1; i >= 0; i--) {
        ans[i] *= suf;
        suf *= nums[i];//suf是后缀积
    }
    return ans;
}
```
#### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

分别计算每个格子的最大前缀(左向右)和最大后缀(右向左)，存两个数组里，然后取前后缀的最小值减去格子的值即为这个格子所能接的水
```c++
int trap(vector<int>& height) {
	//时空复杂度为O(n)
	int n = height.size();
	if (n == 1)return 0;
	vector<int>maxpre(n), maxsuf(n);
	maxpre[0] = height[0];
	maxsuf[n - 1] = height[n - 1];
	for (int i = 1; i < n; i++) {
		maxpre[i] = max(height[i], maxpre[i - 1]);
	}
	for (int j = n - 2; j >= 0; j--) {
		maxsuf[j] = max(height[j], maxsuf[j]);
	}
	int ans = 0;
	for (int i = 0; i < n; i++) {
		ans += min(maxsuf[i], maxpre[i]) - height[i];
	}
	return ans;
}
```

## 双指针
#### [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)
消除两个链表的长度差
```
指针 p 指向 A 链表 指针 q 指向 B 链表 依次往后遍历
如果 p 到了末尾 则 p = headB 继续遍历
如果 q 到了末尾，则 q = headA 继续遍历
比较长的链表指针指向较短链表head时 长度差就消除了
```
```c++
ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
	ListNode* p = headA, * q = headB;
	while (p != q) {
		p = p != nullptr ? p->next : headB;
		q = q != nullptr ? q->next : headA;
	}
	return p;
}
```
#### [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)
```c++
class MedianFinder {
public:
	MedianFinder() :low(us.end()), high(us.end()) {}

	void addNum(int num) {
		int n = us.size();
		us.emplace(num);
		if (!n) {//插入之前为空
			low = high = us.begin();
		}
		else if (n & 1) {//插入之前为奇数
			if (num < *low) low--;
			else high++;
		}
		else {//插入之前为偶数
			if (num > *low && num < *high) {//插在俩中位数中间
				low++;
				high--;
			}
			else if (num >= *high) low++;
			else {
				high--;
				low = high;
			}
		}
	}

	double findMedian() {
		return (*low + *high) / 2.0;
	}
private:
	multiset<int>us;
	multiset<int>::iterator low, high;
};
```
### *相向双指针

#### [167. 两数之和 II - 输入有序数组](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)

有序排列后，如果low+high>target,low之后所有的数与high相加都大于target，high只能舍去(左移)，如果low+high<target,high之前所有的数加上low也肯定都小于target，所以low舍去(右移)

[15. 三数之和](https://leetcode.cn/problems/3sum/)/[18. 四数之和](https://leetcode.cn/problems/4sum/)：枚举第一个数/枚举前两个数，降为两数之和

```c++
vector<int> twoSum(vector<int>& numbers, int target) {
    int i, j;
    vector<int> ans;
    i = 0; j = numbers.size() - 1;
    while (i != j) {
        if (numbers[i] + numbers[j] == target) {
            ans.emplace_back(i+1);
            ans.emplace_back(j+1);
            break;
        }
        else if (numbers[i] + numbers[j] > target) j--;
        else i++;
    }
    return ans;
}
```

#### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

初始化low、high指向数组两端 再分别记录前缀的最大值premax和后缀的最大值sufmax由于前、后缀最大值不会变小 所以当前缀最大值小于后缀最大值的时候 由于短板效应 low位置能接的水已经确定了
```c++
int trap(vector<int>& height) {
	//时间复杂度为O(n) 空间复杂度为O(1)
	int n = height.size();
	int low = 0, high = n - 1;
	int premax = 0, sufmax = 0;
	int ans = 0;
	while (low <= high) {
		premax = max(height[low], premax);
		sufmax = max(height[high], sufmax);
		if (premax >= sufmax) {//右边能接的雨水能够确定下来
			ans += sufmax - height[high];
			high--;
		}
		else {
			ans += premax - height[low];
			low++;
		}
	}
	return ans;
}
```

#### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

```c++
int maxArea(vector<int>& height) {
    int ans = 0;
    int low = 0, high = height.size() - 1;
    while (low < high) {      
        if (height[low] <= height[high]) {
            ans = max(ans, height[low] * (high - low));
            low++;
        }
        else {
            ans = max(ans, height[high] * (high - low));
            high--;
        }
    }
    return ans;
}
```

### *同向双指针 滑动窗口
#### [443. 压缩字符串](https://leetcode.cn/problems/string-compression/)
```c++
int compress(vector<char>& chars) {
	int n = chars.size();
	int write = 0, left = 0;
	string temp;
	for (int read = 0; read < n; read++) {
		if (read == n - 1 || chars[read] != chars[read + 1]) {//读到连续重复字符的末尾
			chars[write++] = chars[read];//记录当前字符
			int num = read - left + 1;//当前字符的个数
			if (num > 1) {
				temp = to_string(num);
				for (const char& t : temp) {
					chars[write++] = t;
				}
			}
			left = read + 1;
		}
	}
	return write;
}
```
#### [1004. 最大连续1的个数 III](https://leetcode.cn/problems/max-consecutive-ones-iii/)
```c++
int longestOnes(vector<int>& nums, int k) {
	int low = 0, high = 0, ans = 0;
	while (high != nums.size()) {
		if (!nums[high]) k--;//遇到0就填
		while (k < 0) {//多填一个 移动左指针
			if (!nums[low]) k++;
			low++;
		}
		ans = max(ans, high - low + 1);//时刻记录最大值
		high++;
	}
	return ans;
}
```
#### [209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)

初始为第一个元素 子数组小于目标则加大右端点来增加子数组长度 若子数组和大于等于目标 缩小左端点 直到右端点为数组最后一个

```c++
int minSubArrayLen(int target, vector<int>& nums) {
    int n = nums.size(), ans = n + 1;
    int low = 0, high = low;
    int sum = nums[low];
    while (high != n) {
        if (sum >= target) {
            sum -= nums[low];
            ans = min(ans, high - low + 1);
            low++;
        }
        else {
            high++;
            if (high != n) 
                sum += nums[high];
        }
    }
    return ans > n ? 0 : ans;
}
```

#### [713. 乘积小于 K 的子数组](https://leetcode.cn/problems/subarray-product-less-than-k/)

```c++
int numSubarrayProductLessThanK(vector<int>& nums, int k) {
	if (k <= 1)return 0; //nums[i]>=1
	int ans = 0, low = 0, high = low;
	int mul = nums[low];//初始化乘积为nums[0]
	while (high != nums.size()) {//固定左端点low 遍历high
		if (mul >= k) {//乘积大于k了 移动左端点
			mul /= nums[low];
			low++;
		}
		else {
			ans += high - low + 1;//新增的！子数组数量
			high++;
			if (high != nums.size()) 
				mul *= nums[high];
		}
	}
	return ans;
}
```
```c++
int numSubarrayProductLessThanK(vector<int>& nums, int k) {
    if (k <= 1)return false;//nums[i]>=1
    int ans = 0, mul = 1, low = 0;
    for (int high = 0; high < nums.size(); high++) {
        mul *= nums[high];
        while(mul >= k) {
            mul /= nums[low];
            low++;
        }
        ans += high - low + 1;//固定了右端点
    }
    return ans;
}
```
#### [100137. 统计最大元素出现至少 K 次的子数组](https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/)
```c++
long long countSubarrays(vector<int>& nums, int k) {//时间复杂度为O(n)
	int m = INT_MIN, cnt = 0;//窗口中m的个数
	for (const int& n : nums) m = max(m, n);
	int low = 0;
	long long ans = 0;
	for (const int& x : nums) {
		if (x == m)cnt++;
		while (cnt == k) {//以x为右端点的子数组恰好合法 这个时候更新k
			if (nums[low] == m)
				cnt--;
			low++;//将low移动到合法子数组的最左端的m的右边(恰好不合法)
		}
		ans += low;//[0,low-1]作为子数组的左端点都是满足条件的
	}
	return ans;
}
```
#### [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

```c++
int lengthOfLongestSubstring(string s) {
    int ans = 0;
    if(s.size()==1)return 1;
    map<char, int> m;
    int low = 0, high = low + 1;
    int num=1;
    m[s[low]]++;
    while (high < s.size()) {
        if (m[s[high]] == 0) {
            m[s[high]]++;
            num++;
            ans = max(ans, num);
            high++;
        }
        else {
            m[s[low]]--;
            num--;
            low++;
        }
    }
    return ans;
}
```
```c++
int lengthOfLongestSubstring(string s) {
    int ans = 0;
    if (s.size() == 1)return 1;
    map<char, int> m;
    int low = 0;
    for (int high = 0; high < s.size(); high++) {
        m[s[high]]++;
        while (m[s[high]] > 1) {
            m[s[low]]--;
            low++;
        }
        ans = max(ans, high - low + 1);
    }
    return ans;
}
```

#### [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)
```c++
vector<int> findAnagrams(string s, string p) {
	int n = p.size();
	vector<int>pcount(26), scount(26);
	vector<int> ans;
	if (n > s.size())return ans;
	for (const char& c : p) pcount[c - 'a']++;
	int i = 0;
	while (i < s.size()) {
		if (i < n - 1) scount[s[i] - 'a']++;
		else {
			scount[s[i] - 'a']++;
			if (scount == pcount)
				ans.emplace_back(i - n + 1);
			scount[s[i - n + 1] - 'a']--;
		}
		i++;
	}
	return ans;
}
```
### *快慢指针

#### [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

判断一个链表是否是有环

```c++
bool hasCycle(ListNode *head) {
    ListNode* slow, * fast;
    slow = fast = head;
    while (slow != nullptr && fast != nullptr) {
        fast = fast->next;
        if (fast != nullptr)
            fast = fast->next;
        else return false;
        slow = slow->next;
        if (slow == fast)
            return true;
    }
    return false;
}
```

#### [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

找到链表中环(至多一个)的入口结点下标(NULL表示无环)

![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/9717c05f-6598-43eb-ac62-e242363fe64a)
```c++
快指针移动距离是慢指针的两倍
=>2(a+b) = a + b + k(b+c)
=>2a + 2b = a + b + b + c + (k - 1)(b + c)
=>a - c = (k - 1)(b + c) 
slow从相遇点出发
head从头结点出发
走c步后 slow在入口
head到入口的距离为a-c 恰好是环长的倍速
继续走 二者相遇处即是入口
```
```c++
ListNode *detectCycle(ListNode *head) {
    ListNode* slow, * fast;
    slow = fast = head;
    while (slow != nullptr && fast != nullptr) {
        fast = fast->next;
        if (fast != nullptr) {
            fast = fast->next;
        }
        else return NULL;
        slow = slow->next;
        if (slow == fast) {
            while (head != slow) {
                head = head->next;
                slow = slow->next;
            }
            return slow;
        }
    }
    return NULL;
}
```
#### [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)
```c++
int findDuplicate(vector<int>& nums) {
	int slow = 0, fast = 0;
	while(slow != fast || (slow == fast && slow == 0)){
	    slow = nums[slow];
	    fast = nums[nums[fast]];
	}
	slow = 0;
	while(slow != fast){
	    slow = nums[slow];
	    fast = nums[fast];
	}
	return slow;
}
```
#### [202. 快乐数](https://leetcode.cn/problems/happy-number/)
```c++
bool isHappy(int n) {
	function<int(int)>get = [&](int x)->int {
		int res = 0;
		while (x) {
			res += (x % 10) * (x % 10);
			x /= 10;
		}
		return res;
	};
	int slow = get(n), fast = get(get(n));
	while (fast != 1 && fast != slow) {
		slow = get(slow);
		fast = get(get(fast));
	}
	return fast == 1;
}
```
### *前后指针

#### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

```c++
ListNode* List::removeNthFromEnd(ListNode* head, int n) {
	ListNode* dummy = new ListNode(head);
	ListNode* front, * back;
	front = back = dummy;
	while (n != 0) {
		front = front->next;
		n--;
	}
	while (front->next != nullptr) {
		front = front->next;
		back = back->next;
	}
	back->next = back->next->next;
	return dummy->next;
}
```

#### [83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)

删除有序链表中的多余的重复结点

```c++
ListNode* deleteDuplicates(ListNode* head) {
        if(head==nullptr)return head;
 		ListNode* front = head->next;
        ListNode* back = head;
        while (front != nullptr) {
            if (front->val != back->val) {
                back->next = front;
                back = front;
            }
            front = front->next;
        }
        back->next = nullptr;
        return head;
}
```

[82. 删除排序链表中的重复元素 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)删除有序链表中重复出现的结点，仅剩下没有重复的结点

```c++
ListNode* deleteDuplicates(ListNode* head) {
    if (head == nullptr)return head;
        ListNode * pre, * front, * back;
        ListNode* dummy=new ListNode(0,head);
        pre = dummy;
        back = dummy->next;
        front = back->next;
        while (front != nullptr) {
            if (front->val != back->val) {
                if (back->next->val != back->val) {
                    pre->next = back;
                    pre = back;
                }
                back = front;
            }
            front = front->next;
        }
        if (back->next == nullptr) {
            pre->next = back;
        }
        else {
            pre->next = nullptr;
        }
        return dummy->next;
}
```

## 回溯

```
用一个path记录路径上的选择
回溯三问：
1.当前的操作？枚举path[i]的选择

2.子问题？构造>=i的部分

3.下一个子问题？构造>=i+1的部分
```

```c++
模板一：输入的视角(选不选)
dfs(i)->dfs(i+1)
```
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/37100645-939f-4ece-b679-754e2bcfaff1)

```
模板二：答案的视角(枚举)
```
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/e3773247-e287-4270-a8b7-543b889da942)
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/71781975-58d6-45f0-a614-c71870672fd4)

```
排列型模板:数组path记录路径上的数(已选数字)集合s记录剩余未选数字
```
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/f00d3275-304e-4490-a7eb-0e7e03ccbeba)

### *子集型

#### [78. 子集](https://leetcode.cn/problems/subsets/)

```c++
//模板一
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>>ans;
    vector<int>path;
    int n = nums.size();
    std::function<void(int)> dfs = [&](int i) {
        if (i == n) {
            ans.push_back(path);
            return;
        }
        //不选nums[i]
        dfs(i + 1);
        //选nums[i]
        path.push_back(nums[i]);
        dfs(i + 1);
        path.pop_back();//恢复现场！
    };
    dfs(0);
    return ans;
}
```

```c++
 //模板二
 vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>>ans;
    vector<int>path;
    int n = nums.size();
    std::function<void(int)> dfs = [&](int i) {
        ans.push_back(path);
        //if (i == n)return;
        for (int j = i; j < n; j++) {
            path.push_back(nums[j]);
            dfs(j + 1);
            path.pop_back();//恢复现场！
        }
    };
    dfs(0);
    return ans;
}
```

### *分割型

#### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

```c++
vector<string> letterCombinations(string digits) {
    vector<string>a = { "","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz" };
    vector<string>ans;
    int n = digits.size();
    if (digits.empty())return ans;
    string path;
    path.resize(n);
    std::function<void(int)> dfs = [&](int i) {
        if (i == n) {
            ans.push_back(path);
            return;
        }
        for (const auto& c : a[int(digits[i]) - 48]) {
            path[i] = c;
            dfs(i + 1);
        }
    };
    dfs(0);
    return ans;
}
```

#### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

```c++
vector<vector<string>> partition(string s) {
	vector<vector<string>>ans;
	vector<string>path;
	int n = s.size();
	function<bool(int, int)>is = [&](int low, int high)->bool {
		while (low < high) {
			if (s[low] != s[high])return false;
			low++; high--;
		}
		return true;
	};
	function<void(int)>dfs = [&](int index) {
		if (index >= n) {
			ans.emplace_back(path);
			return;
		}
		for (int i = index; i < n; i++) {
			if (is(index, i)) {
				path.emplace_back(s.substr(index, i - index + 1));
				dfs(i + 1);
				path.pop_back();//恢复现场
			}
		}
	};
	dfs(0);
	return ans;
}
```

### *组合型
组合数学公式 C(n,k)= n!/(k! * (n-k)!)  组合无序 因此要除以 k!

剪枝技巧： 逆序枚举

```
设path长为m
那么还需选d=k-m个数
设当前要从[1,i]这i个数中选数
如果i<d最后必然没法选够k个数 不需要继续递归
```

#### [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

从1-9中选择k个数(最多选一次)使其和为n

```c++
 //模板二+剪枝技巧：
vector<vector<int>> combinationSum3(int k, int n) {
    vector<vector<int>>ans;
	vector<int>path;
	int sum = 0;
	std::function<void(int)> dfs = [&](int i) {
		if (path.size() == k && sum == n) {
			ans.push_back(path);
			return;
		}
		if (9 - i < k - path.size())
			return;
		//枚举当前可以选的数
		for (int j = i+1; j < 10; j++) {
			path.push_back(j);
			sum += j;
			dfs(j);
			//恢复现场
			path.pop_back();
			sum -= j;
		}
	};
	dfs(0);
	return ans;
}
```

#### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

从一个数组中选取一些可重复数使得其和为target，返回所有组合

```c++
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
	vector<vector<int>>ans;
    vector<int>path;
    int sum = 0;
    sort(candidates.begin(), candidates.end());
    int n = candidates.size();
    std::function<void(int)> dfs = [&](int i) {
        if (i == n || sum > target)return;
        if (sum == target) {
            ans.push_back(path);
            return;
        }
        if (candidates[i] > target)return;
        //不选
        dfs(i + 1);
        //选
        sum += candidates[i];
        path.push_back(candidates[i]);
        dfs(i);
        sum -= candidates[i];
        path.pop_back();	
    };
    dfs(0);
    return ans;
}
```

#### [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

从一个元素数组中选取一些元素组合(组合中每个数组元素只使用一次，但是有多个元素值相同)使得其和为target，返回所有组合

```c++
vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    vector<vector<int>>ans;
    vector<int>path;
    int sum = 0;
    map<int, int> m;
    vector<int> nums;
    for (const auto& c : candidates) {
        m[c]++;
    }
    for (const auto& v : m) {
        nums.push_back(v.first);
    }
    int n = nums.size();
    std::function<void(int)> dfs = [&](int i) {
        if (sum > target)return;
        if (sum == target) {
            ans.push_back(path);
            return;
        }
        if (i == n)return;
        if(nums[i]>target)return;
        //不选
        dfs(i + 1);
        //选 /选k次
        for (int k = 1; k <= m[nums[i]]; k++) {
            sum += nums[i] * k;
            if (sum > target) {
                sum -= nums[i] * k;
                break; 
            }
            for (int kk = 0; kk < k; kk++) {
                path.push_back(nums[i]);
            }
            dfs(i + 1);
            sum -= nums[i] * k;
            for (int kk = 0; kk < k; kk++) {
                path.pop_back();
            }
        }
    };
    dfs(0);
    return ans;
}
```

### *排列型
排列数学公式 A(n,k)= n!/(n-k)!

不同于组合型，(i,j)!=(j,i)
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/51adcf91-e69c-466d-839f-1fb280e7734d)

时间复杂度为所有叶子结点到根节点的路径和O(n*n!)

#### [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

给一个含重复数字的数组，给出它所有的不重复的全排列

```c++
vector<vector<int>> permuteUnique(vector<int>& nums) {
    //时间复杂度O(n!)	
    vector<vector<int>>ans;
    vector<int>path;
    int n = nums.size();
    vector<int>on_path(n, 0);
    std::function<void(int)> dfs = [&](int i) {
        if (i == n) {
            ans.push_back(path);
            return;
        }
        for (int j = 0; j < n; j++) {
            if (on_path[j] == 0) {
                path.push_back(nums[j]);
                on_path[j] = 1;
                dfs(i + 1);
                path.pop_back();
                on_path[j] = 0;
            }
        }
    };
    dfs(0);
    sort(ans.begin(),ans.end());
    auto end = unique(ans.begin(), ans.end());
    ans.erase(end, ans.end());
    return ans;
}
```

#### [51. N 皇后](https://leetcode.cn/problems/n-queens/)

```c++
vector<vector<string>> solveNQueens(int n) {
	vector<string> temp(n, string(n, '.'));//初始化棋盘
	vector<vector<string>>ans;
	vector<int>col(n);//表示该行在第几列放入
	vector<bool>on_col(n, false);//表示这列是否放入
	std::function<bool(int, int)> isok =
		[&](int x, int y)->bool {//判断对角线
		//判断左上 行号-列号相同 右上 行号+列号相同
		for (int r = 0; r < x; r++) {
			int c = col[r];
			if (x + y == r + c || x - y == r - c)
				return false;
		}
		return true;
	};
	std::function<void(int)> dfs = [&](int i) {//递推行
		if (i == n) {
			for (int k = 0; k < n; k++) {//画出该棋盘
				temp[k][col[k]] = 'Q';
			}
			ans.push_back(temp);
			for (int k = 0; k < n; k++) {//恢复棋盘
				temp[k][col[k]] = '.';
			}
		}
		for (int j = 0; j < n; j++) {
			if (!on_col[j] && isok(i, j)) {//该列没插入过且对角线满足
				col[i] = j;
				on_col[j] = true;
				dfs(i + 1);
				on_col[j] = false;//恢复现场
			}
		}
	};
	dfs(0);
	return ans;
}
```
### *DFS
#### [332. 重新安排行程](https://leetcode.cn/problems/reconstruct-itinerary/)
```c++
vector<string> findItinerary(vector<vector<string>>& tickets) {
	int n = tickets.size();
	sort(tickets.begin(), tickets.end(),
		[&](const vector<string>& a, const vector<string>& b) {
			return a[1] < b[1]; });//先将票按字典序排好
	unordered_map<string, vector<pair<string, bool>>>um;//键表示起点 vector存终点以及该票是否没用过
	for (const auto& t : tickets) {
		um[t[0]].emplace_back(t[1], true);
	}
	vector<string>path; path.emplace_back("JFK");
	function<bool(string)>dfs = [&](string s)->bool {
		if (path.size() > n) return true;
		string last = "";//票有重复的 防止重复计算回溯 不超时的关键步骤
		for (auto& ne : um[s]) {
			if (ne.second && ne.first != last) {
				last = ne.first;
				path.emplace_back(ne.first);
				ne.second = false;
				if (dfs(ne.first))return true;
				path.pop_back();
				ne.second = true;
			}
		}
		return false;
	};
	dfs("JFK");
	return path;
}
```
#### [37. 解数独](https://leetcode.cn/problems/sudoku-solver/)
```c++
void solveSudoku(vector<vector<char>>& board) {
	vector<bitset<9>>rows(9, bitset<9>(0)), cols(9, bitset<9>(0));
	vector<vector<bitset<9>>>cells(3, vector<bitset<9>>(3, bitset<9>(0)));
	function<vector<int>()>next = [&]()->vector<int> {//下一个最少回溯的位置
		vector<int>ret(2); int min_ = 10, cunt;
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (board[i][j] == '.') {
					cunt = (~(rows[i] | cols[j] | cells[i / 3][j / 3])).count();
					if (cunt < min_) {
						ret = { i,j };
						min_ = cunt;
					}
				}
			}
		}
		return ret;
	};
	function<bool(int)>dfs = [&](int cnt)->bool {
		if (!cnt)return true;
		auto it = next();
		auto status = ~(rows[it[0]] | cols[it[1]] | cells[it[0] / 3][it[1] / 3]);
		for (int i = 0; i < 9; i++) {//(it[0],it[1])的位置
			if (status.test(i)) {//当前位置的值为i + '1'
				rows[it[0]][i] = cols[it[1]][i] = cells[it[0] / 3][it[1] / 3][i] = true;
				board[it[0]][it[1]] = i + '1';
				if (dfs(cnt - 1))return true;//剪枝
				board[it[0]][it[1]] = '.';
				rows[it[0]][i] = cols[it[1]][i] = cells[it[0] / 3][it[1] / 3][i] = false;
			}
		}
		return false;
	};
	int cnt = 0, n;
	for (int i = 0; i < 9; i++) {//init
		for (int j = 0; j < 9; j++) {
			cnt += (board[i][j] == '.');
			if (board[i][j] != '.') {
				n = board[i][j] - '1';
				rows[i] |= (1 << n);
				cols[j] |= (1 << n);
				cells[i / 3][j / 3] |= (1 << n);
			}
		}
	}
	dfs(cnt);
}
```
## 动态规划

定义状态 状态转移方程 时间复杂度：状态的个数*计算状态的时间

回溯/递归+记忆化搜索->动态规划  记忆化搜索->循环递推

[2008. 出租车的最大盈利](https://leetcode.cn/problems/maximum-earnings-from-taxi/)
```c++
/*dfs(i) 表示从1到i可以赚的最多的钱
当没人在i下车的时候dfs(i) = dfs(i−1)
有人在i下车的时候 枚举以i为终点的起点s 计算子问题dfs(s)*/
long long maxTaxiEarnings(int n, vector<vector<int>>& rides) {
	vector<vector<pair<int, int>>>v(n + 1);
	for (const auto& r : rides) {//v[end]=vector{pair(star,earning)}
		v[r[1]].emplace_back(r[0], r[1] - r[0] + r[2]);
	}
	//递归+记忆化搜索
	vector<long long>dp(n + 1, -1);
        dp[0] = 0;
        function<long long(int)>dfs = [&](int i)->long long {
            if (dp[i] != -1)return dp[i];
            if (v[i].size() == 0) {
                dp[i] = dfs(i - 1);
                return dp[i];
            }
            dp[i] = dfs(i - 1);
            for (const auto& vv : v[i]) {
                dp[i] = max(vv.second + dfs(vv.first), dp[i]);
            }
            return dp[i];
        };
        return dfs(n);
	//递推
	vector<long long>dp(n + 1, 0);
	for (int i = 1; i <= n; i++) {
		dp[i] = dp[i - 1];
		for (const auto& vv : v[i]) {
			dp[i] = max(dp[i], vv.second + dp[vv.first]);
		}
	}
	return dp[n];
}
```
[2830. 销售利润最大化](https://leetcode.cn/problems/maximize-the-profit-as-the-salesman/)
### *打家劫舍

[198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

```
dfs(i)=max(dfs(i-1),dfs(i-2)+nums[i])
```

选：nums[i]+进入dfs(i-2)(从最后一个开始选) 不选进入上一层

从后向前算：记忆化搜索递归

普通回溯

![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/cc18f256-6184-49a5-892a-23f1b1051c52)

记忆化搜索后的回溯

![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/f2a23cc9-f1de-4b68-8e1e-0320e17c70f8)
```c++
int rob(vector<int>& nums) {
    //时间复杂度=空间复杂度=O(n)
    int n = nums.size();
	vector<int>path(n, -1);//记录dfs[i]的值 优化时间
	std::function<int(int)> dfs = [&](int i)->int {
		if (i <0) {
			return 0;
		}
		if (path[i] != -1)return path[i];
		int ans = max(dfs(i - 1), nums[i] + dfs(i - 2));
		path[i] = ans;
		return ans;
	};
	return dfs(n - 1);
}
```
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/22c30b1d-d6b7-4f30-bd39-7b3ad23c4225)
1：1 dfs->f数组循环递推

```c++
 int rob(vector<int>& nums) {
    //时间复杂度=O(n) 空间复杂度=O(n)
    int n = nums.size();
    vector<int>path(n + 2, 0);
    for (int i = 0; i < nums.size(); i++) {
        path[i + 2] = max(path[i + 1], path[i] + nums[i]);
    }
    return path[n + 1];
}
```

空间复杂度优化为O(1) f数组循环递推->滚动数组

```c++
 int rob(vector<int>& nums) {
    //时间复杂度=O(n) 空间复杂度=O(1)
	int n = nums.size();
	int f0, f1,f;
	f0 = f1 = f = 0;
	for (const auto& v : nums) {
		f = max(f1, f0 + v);
		f0 = f1;
		f1 = f;
	}
	return f1;
}
```
针对环形房子的打家劫舍 只要对[0,n-1)和[1,n) 两个区间求两次 取最大即可
### *0/1背包

```
0/1背包：有n个物品  第i个物品的体积为w[i]  价值为v[i]
每个物品至多选一个  求体积和不超过capacity时的最大价值和
dfs(i,c)=max(dfs(i-1,c),dfs(i-1,c-w[i])+v[i])
```

```c++
//0-1背包问题 记忆化搜索数组递归版
std::function<int(int capacity, vector<int> w, vector<int>v)> zero_one_knapsack =
[](int capacity, vector<int> w, vector<int>v)->int {
	int n = w.size();
	vector<vector<int>>path(n, vector<int>(n, -1));//记忆化:使用二维数组表示两个参数
	std::function<int(int, int)> dfs = [&](int i, int c)->int {
		if (i < 0)return 0;
		if (c < w[i]) {//物体体积大于剩余容量，只能不选
			if (path[i - 1][c] != -1) return path[i - 1][c];
			else {
				path[i - 1][c] = dfs(i - 1, c);
				return path[i - 1][c];
			}
		}
		if (path[i - 1][c] == -1)path[i - 1][c] = dfs(i - 1, c);
		if (path[i - 2][c - w[i]] == -1)path[i - 2][c - w[i]] = dfs(i - 2, c - w[i]);
		return max(path[i - 1][c], path[i - 2][c - w[i]] + v[i]);
	};
	return dfs(n - 1, capacity);
};
```

#### [494. 目标和](https://leetcode.cn/problems/target-sum/)

在非负整数数组 nums的每个数前加+/-使得数组和为target

```c++
//记忆化搜索
//dfs(i,c)=dfs(i-1,c)+dfs(i-1,c-w[i])
int findTargetSumWays(vector<int>& nums, int target) {
    //所有前面为+号的数之和为p
    //所有数之和为s
    //所有前面为-号的数之和为s-p
    //-(s-p)+p=target => p=(s+t)/2 => s+t为非负偶数
    for (const auto& v : nums) {//target=s+t
        target += v;
    }
    if (target < 0 || target % 2 == 1)return 0;
    target /= 2;
    int n = nums.size();
    //记忆化搜搜
    vector<vector<int>> cache(n, vector<int>(target+1, -1));
    //dfs(i,j)表示从前i个数中选出恰好为j的方案数 j取值[0,target]故记忆化target+1
    std::function<int(int, int)> dfs = [&](int i, int j)->int {
        if (i < 0) {
            if (j == 0)return 1;
            else return 0;
        }
        int& res = cache[i][j];
        if (res != -1)return res;
        if (j < nums[i]) {//物体体积大于剩余容量，只能不选
            res = dfs(i - 1, j);
            return res;
        }
        else {
            res = dfs(i - 1, j) + dfs(i - 1, j - nums[i]);
            return res;
        }
    };
    return dfs(n - 1, target);
}
```

```c++
//递推
//f[i][c]=f[i-1][c]+f[i-1][c-w[i]]
//f[i+1][c]=f[i][c]+f[i][c-w[i]]
//数组初始值为边界条件
vector<vector<int>> f(n + 1, vector<int>(target + 1, 0));
f[0][0] = 1;//边界条件作为初始值
for (int i = 0; i < n; i++) {
    for (int j = 0; j < target+1; j++) {
        if (j < nums[i]) {
            f[i + 1][j] = f[i][j];//只能不选
        }
        else {
            //前面 不选的方案数+选的方案数
            f[i + 1][j] = f[i][j] + f[i][j - nums[i]];
        }
    }
}
return f[n][target];
```

```c++
//滚动数组递推
vector<vector<int>> f(2, vector<int>(target + 1, 0));
f[0][0] = 1;//边界条件作为初始值
for (int i = 0; i < n; i++) {
    for (int j = 0; j < target+1; j++) {
        if (j < nums[i]) {
            f[(i + 1)%2][j] = f[i%2][j];//只能不选
        }
        else {
            //前面 不选的方案数+选的方案数
            f[(i + 1) % 2][j] = f[i % 2][j] + f[i % 2][j - nums[i]];
        }
    }
}
return f[n % 2][target];
```

```c++
 //一个数组递推
vector<int> f(target + 1, 0);
f[0] = 1;//边界条件作为初始值
for (int i = 0; i < n; i++) {
    //从后向前遍历
    for (int j = target; j >= nums[i]; j--) {
        f[j] += f[j - nums[i]];
    }
}
return f[target];
```

```
常见变形：
1、至多装capacity 求方案数/最大价值和
2、恰好装capacity 求方案数/最大/最小价值和
3、至少装capacity 求方案数/最小价值和
```

判断是否能恰好装满背包
#### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)
```c++
bool canPartition(vector<int>& nums) {
	int n = nums.size();
	if (n == 1)return false;
	int target = 0, Max = 0;
	for (const int& v : nums) {
		target += v;
		Max = max(Max, v);
	}
	if (target & 1)return false;
	target /= 2;
	//求和为target的子序列 => 0/1背包 恰好装capacity 
	if (Max > target) return false;
	vector<vector<int>> dp(n, vector<int>(target + 1, 0));
	for (int i = 0; i < n; i++) {
		dp[i][0] = true;
	}
	dp[0][nums[0]] = true;
	for (int i = 1; i < n; i++) {
		for (int j = 1; j <= target; j++) {
			if (j >= nums[i]) {//可以装
				dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num];
			}
			else { //装不下
				dp[i][j] = dp[i - 1][j];
			}
		}
	}
	return dp[n - 1][target];
}
```
### *完全背包

```
完全背包：有n个物品  第i个物品的体积为w[i]  价值为v[i]
每个物品无限次重复选  求体积和不超过capacity时的最大价值和
dfs(i,c)=max(dfs(i-1,c),dfs(i,c-w[i])+v[i])
```

#### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)

数组 coins 元素表示不同面额的硬币 返回使用最少的硬币达到 amount

```c++
int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<int>dp(amount + 1, INT_MAX);
        dp[0] = 0;
        for (int i = 1; i <= amount; ++i) {
            for (int j = 0; j < n; ++j) {
                if (coins[j] <= i && dp[i - coins[j]] != INT_MAX) {
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1);
                }
            }
        }
        return dp[amount] == INT_MAX ? -1 : dp[amount];
    }
```
类似题目[279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)
### *组合型

#### [1155. 掷骰子等于目标和的方法数](https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/)

```c++
int numRollsToTarget(int n, int k, int target) {
    const int mod = 1000000007;//数据太大，取模
    int ans = 0;
    //dp[i][j]表示第i步到达j的数量
    vector<vector<int>>dp(31, vector <int>(1001, 0));
    for (int t = 1; t <= k; t++) {//边界条件
        dp[1][t] = 1;
    }
    for (int i = 2; i <= n; i++) {
        for (int j = i; j <= 1000 && j <= i * k; j++) {
            for (int p = 1; p <= k; p++) {
                if (j >= p) {//转移方程dp[i][j]+=dp[i-1][j-(1-k)]
                    dp[i][j] = (dp[i][j] + dp[i - 1][j - p]) % mod;
                }
                else break;
            }
        }
    }
    return dp[n][target];
}
```

### *子序列

#### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

```c++
//记忆化搜索
int longestCommonSubsequence(string text1, string text2) {
    int l1 = text1.size();
    int l2 = text2.size();
    vector<vector<int>>path(l1, vector<int>(l2, -1));
    std::function<int(int, int)>dfs = [&](int i, int j)->int {
        if (i < 0 || j < 0)return 0;
        if (path[i][j] != -1)return path[i][j];
        if (text1[i] == text2[j]) {
            path[i][j] = dfs(i - 1, j - 1) + 1;
            return path[i][j];
        }
        else {
            path[i][j] = max(dfs(i - 1, j), dfs(i, j - 1));
            return path[i][j];
        }
    };
    return dfs(l1 - 1, l2 - 1);
```

```c++
//递推
int longestCommonSubsequence(string text1, string text2) {
    int l1 = text1.size();
    int l2 = text2.size();
    vector<vector<int>>f(l1+1, vector<int>(l2+1, 0));
    for (int i = 0; i < l1; i++) {
        for (int j = 0; j < l2; j++) {
            if (text1[i] == text2[j]) {
                f[i + 1][j + 1] = f[i][j] + 1;
            }
            else {
                f[i + 1][j + 1] = max(f[i][j + 1], f[i + 1][j]);
            }
        }
    }
    return f[l1][l2];
}
```

#### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

将nums排序去重以后与原来的nums求一个最长公共子序列(LCS)就可以得到nums的最长递增子序列(LIS)

```c++
//记忆化搜索
int lengthOfLIS(vector<int>& nums) {
	//时间复杂度O(n^2) 空间复杂度O(n)
	int n = nums.size();
	//dfs(i)表示以第i个位置结尾的LIS
	vector<int>path(n, -1);//记忆化搜索
	function<int(int)>dfs = [&](int i)->int {
		if (path[i] != -1)return path[i];
		int res = 0;
		for (int j = 0; j < i; j++) {//枚举i之前的数
			if (nums[j] < nums[i]) {
				res = max(res, dfs(j));
			}
		}
		path[i] = res + 1;
		return res + 1;
	};
	int ans = 0;
	for (int i = 0; i < n; i++) {
		ans = max(ans, dfs(i));
	}
	return ans;
}
```

```c++
int lengthOfLIS(vector<int>& nums) {
	//递推
	//时间复杂度O(n^2) 空间复杂度O(n)
	int n = nums.size();
	vector<int>path(n, 0);//数组递推
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			if (nums[j] < nums[i]) {
				path[i] = max(path[i], path[j]);
			}
		}
		path[i] += 1;
	}
	int ans = 0;
	for (const auto& v : path) {
		ans = max(ans, v);
	}
	return ans;
}
```

#### [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)

```c++
int minDistance(string word1, string word2) {
	int l1 = word1.size();
	int l2 = word2.size();
	vector<vector<int>>path(l1, vector<int>(l2, -1));
	std::function<int(int, int)>dfs =
		[&](int i, int j)->int {
		if (i < 0) return j + 1;
		else if (j < 0)return i + 1;
		if (path[i][j] != -1)return path[i][j];
		if (word1[i] == word2[j]) {
			path[i][j] = dfs(i - 1, j - 1);
			return path[i][j];
		}
		else {
			path[i][j] = min(min(dfs(i - 1, j), dfs(i, j - 1)), dfs(i - 1, j - 1)) + 1;
			return path[i][j];
		}
	};
	return dfs(l1 - 1, l2 - 1);
}
```

#### [583. 两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/)
```c++

```
### *子数组、子串
思考子数组、子串统计类问题的通用技巧:

将所有子串按照其末尾字符的下标分组

考虑两组相邻的子串：以 s[ i−1 ] 结尾的子串、以 s[ i ] 结尾的子串

以 s[ i ] 结尾的子串，可以看成是以 s[ i−1 ] 结尾的子串，在末尾添加上 s[ i ] 组成

pre简单题[53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)
#### [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)
```c++
int maxProduct(vector<int>& nums) {
	//dpmax[i]为下标i结尾的乘积最大子数组 优化空间为dpmax(以下标i-1为结尾的最大子数组)
	//dpmin[i]为下标i结尾的乘积最小子数组 优化空间为dpmin(以下标i-1为结尾的最小子数组)
	int n = nums.size();
	int dpmax, dpmin, ans;
	dpmax = dpmin = ans = nums[0];
	for (int i = 1; i < n; i++) {
		int fmax = max(nums[i], max(nums[i] * dpmax, nums[i] * dpmin));
		int fmin = min(nums[i], min(nums[i] * dpmax, nums[i] * dpmin));
		dpmax = fmax; dpmin = fmin;
		ans = max(ans, fmax);
	}
	return ans;
}
```
#### [647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/)
```c++
int countSubstrings(string s) {
	int n = s.size(), ans = 0;
	vector<vector<bool>>dp(n, vector<bool>(n, false));
	for (int i = n - 1; ~i; i--) {//从dp[i + 1][j - 1]->dp[i][j] i从最大处n - 1开始
		for (int j = i; j < n; j++) {//dp[i][j]表示从下标i到下标j的子串
			if (s[i] == s[j]&&(j - i <= 1 || dp[i + 1][j - 1])) {
				ans++;
				dp[i][j] = true;	
			}
		}
	}
	return ans;
}
```
#### [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)
```c++

```

### *状态机DP
#### [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/) (不限交易次数)

从最后一天开始思考：从第0天到第i天结束时的利润 = 从第0天到第i-1天结束时的利润 + 第i天的利润

第i天的利润为(0:什么都不做 状态不变;prices[i]:卖掉股票 状态变成未持有;-prices[i]:买入股票 状态变成持有)
```c++
//记忆化搜索
int maxProfit(vector<int>& prices) {
	int n = prices.size();
	vector<vector<int>>memo(n, vector<int>(2,-1));//记忆化搜索
	//dfs(i,hold) 表示第i天结束时(也就是第i+1天开始时)是否拥有股票
	function<int(int, bool)>dfs = [&](int i, bool hold)->int {
		if (i < 0) {//边界条件 第0天开始有股票不合法设为-无穷 未持有股票 利润为0
			return hold ? INT_MIN : 0;
		}
		if (memo[i][hold] != -1)return memo[i][hold];
		if (hold) {//第i天结束拥有为max(第i-1天持有但什么都不做,第i-1天未持有买入)
			memo[i][hold] = max(dfs(i - 1, 1), dfs(i - 1, 0) - prices[i]);
			return memo[i][hold];
		}
		memo[i][hold] = max(dfs(i - 1, 0), dfs(i - 1, 1) + prices[i]);
		return memo[i][hold];
	};
	return dfs(n - 1, 0);
}
//递推
int maxProfit(vector<int>& prices) {
	int n = prices.size();
	vector<vector<int>>f(n+1, vector<int>(2,0));
	f[0][0] = 0; f[0][1] = INT_MIN;
	for (int i = 0; i < n; i++) {
		f[i + 1][0] = max(f[i][0], f[i][1] + prices[i]);
		f[i + 1][1] = max(f[i][1], f[i][0] - prices[i]);
	}
	return f[n][0];
}
//滚动数组
int maxProfit(vector<int>& prices) {
	int f0 = 0, f1 = INT_MIN;
	for (const int &p: prices) {
	    //计算f1还需要原来的f0，不能直接将f0覆盖掉 先用new_f0记录
	    int new_f0 = max(f0, f1 + p);
	    f1 = max(f1, f0 - p);
	    f0 = new_f0;
	}
	return f0;
}
```
[309. 买卖股票的最佳时机含冷冻期](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/)
```c++
//同122题 仅需修改持有的时候 是从第i-2天未持有转移过来的即可 类比打家劫舍
if (hold) {//第i天结束拥有为max(第i-1天持有但什么都不做,第i-2天未持有买入)
	memo[i][hold] = max(dfs(i - 1, 1), dfs(i - 2, 0) - prices[i]);
	return memo[i][hold];
}
```

#### [188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)  (至多交易k次)
```c++
//记忆化搜索
int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        vector<vector<vector<int>>>memo(n, vector<vector<int>>(k+1, vector<int>(2, -1)));//记忆化搜索
        //dfs(i,hold) 表示第i天结束时(也就是第i+1天开始时)是否拥有股票
        function<int(int, int, bool)>dfs = [&](int i, int j, bool hold)->int {
            if (j < 0) {
                return INT_MIN;
            }
            if (i < 0) {//边界条件 第0天开始有股票不合法设为-无穷 未持有股票 利润为0
                return hold ? INT_MIN : 0;
            }
            if (memo[i][j][hold] != -1)return memo[i][j][hold];
            if (hold) {//第i天结束拥有为max(第i-1天持有但什么都不做,第i-1天未持有买入)
                memo[i][j][hold] = max(dfs(i - 1,j, 1), dfs(i - 1,j, 0) - prices[i]);//这里或下面都可以改成j-1
                return memo[i][j][hold];
            }
            memo[i][j][hold] = max(dfs(i - 1,j, 0), dfs(i - 1,j-1, 1) + prices[i]);//这里改成j-1
            return memo[i][j][hold];
        };
        return dfs(n - 1, k, 0);
}
//改成递推+空间优化
int maxProfit(int k, vector<int>& prices) {
	vector<vector<int>>f(k + 2, vector<int>(2, -1200));
	for (int j = 1; j <= k + 1; j++) {
	    f[j][0] = 0;
	}
	for (int p: prices) {
	    for (int j = k + 1; j > 0; j--) {
		f[j][0] = max(f[j][0], f[j][1] + p);
		f[j][1] = max(f[j][1], f[j - 1][0] - p);
	    }
	}
	return f[k + 1][0];
}
```

### *区间DP
#### [516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/)

方法一:将s逆置 求与原来的最长公共子序列的长度 即为答案
方法二: 从两侧向内缩小问题规模 选或不选 (判断s的第一个字符和最后一个字符是否相等 相等的话都选上 不相等的话 变成选第一个 还是选最后一个) 
```c++
//方法二 记忆化搜索版 时空复杂度O(n^2) 
int longestPalindromeSubseq(string s) {
	int n = s.size();
	vector<vector<int>>memo(n, vector<int>(n, -1));
	function<int(int, int)>dp = [&](int i, int j)->int {
		if (i == j) {
			return 1;
		}
		if (i > j)return 0;
		if (memo[i][j] != -1)return memo[i][j];
		if (s[i] == s[j]) {
			memo[i][j] = dp(i + 1, j - 1) + 2;
			return memo[i][j];
		}
		else {
			memo[i][j] = max(dp(i + 1, j), dp(i, j - 1));
			return memo[i][j];
		}
	};
	return dp(0, n - 1);
}
```
#### [1039. 多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/) 

分割成多个规模更小的子问题 枚举选哪个
```c++
int minScoreTriangulation(vector<int>& values) {
	//时间复杂度O(n^3) 空间复杂度O(n^2)
	int n = values.size();
	vector<vector<int>>memo(n, vector<int>(n, -1));
	function<int(int, int)> dfs = [&](int i, int j)->int {
		if (i + 1 == j) {
			return 0;
		}
		if (memo[i][j] != -1)return memo[i][j];
		int res = INT_MAX;
		for (int k = i + 1; k < j; k++) {//枚举第三个顶点k
			//res=(左边三角形的分数 + 右边三角形的分数 + 当前三角形的分数)min
			res = min(res, dfs(i, k) + dfs(k, j) + values[i] * values[j] * values[k]);
		}
		memo[i][j] = res;
		return memo[i][j];
	};
	return dfs(0, n - 1);//固定一条以顶点0和顶点n-1两个顶点构成的边 枚举第三个顶点
}
//递推
//时间复杂度O(n^3) 空间复杂度O(n^2)
int n = values.size();
vector<vector<int>>f(n, vector<int>(n, 0));
// f[i][j]=(f[i][k]+f[k][j]+v[i]*v[j]*v[k])min
// i < k < j 故i倒序枚举 j正序枚举
for (int i = n - 3; i >= 0; i--) {
	for (int j = i + 2; j < n; j++) {
		int res = INT_MAX;
		for (int k = i + 1; k < j; k++) {//枚举第三个顶点k
			//res=(左边三角形的分数 + 右边三角形的分数 + 当前三角形的分数)min
			res = min(res, f[i][k] + f[k][j] + values[i] * values[j] * values[k]);
		}
		f[i][j] = res;
	}
}
return f[0][n - 1];
```
#### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)
```c++
int longestValidParentheses(string s) {
	int ans = 0, n = s.size();
	vector<int> dp(n);//dp[i]表示以下标i作为子串的终点
	for (int i = 1; i < n; ++i) {
		if (s[i] == ')') {//子串的终点只能是')'
			if (s[i - 1] == '(') {//情况 ()
				dp[i] = i >= 2 ? dp[i - 2] + 2 : 2;
			}
			else if (dp[i - 1] > 0) {//情况 )) 
				if (i - 1 - dp[i - 1] >= 0
					&& s[i - 1 - dp[i - 1]] == '(') {//情况x(...)) x的下标>=0&&x为(
					dp[i] = dp[i - 1] + 2;  //dp[i]至少为 ((...)) dp[i-1]=(...)
					if (i - 2 - dp[i - 1] >= 0) { //情况x((...)) 前面还有x
						dp[i] += dp[i - 2 - dp[i - 1]];
					}
				}
			}
		}
		ans = max(ans, dp[i]);
	}
	return ans;
}
```
### *树形DP

二叉树 边权型
#### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)
```c++
int diameterOfBinaryTree(TreeNode* root) {
	//时空复杂度都是O(n)
	int ans = 0;
	function<int(TreeNode*)> dfs = [&](TreeNode* root)->int {
		if (root == nullptr)return 0;//节点为空与父节点没有边
		int l_len = dfs(root->left);//左子树的高度
		int r_len = dfs(root->right);//右子树的高度
		ans = max(ans, l_len + r_len);//计算所有节点左子树高度+右子树高度的最大值
		return max(l_len, r_len) + 1;//加1表示这个非空节点与父节点有一个边 高度+1
	};
	dfs(root);
	return ans;
}
```
二叉树 点权型
#### [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)
```c++
int maxPathSum(TreeNode* root) {
	//时空复杂度都是O(n)
	int ans = INT_MIN;
	function<int(TreeNode*)> dfs = [&](TreeNode* root)->int {
		if (root == nullptr)return 0;
		int left_num = dfs(root->left);
		int right_num = dfs(root->right);
		ans = max(ans, root->val + left_num + right_num);//当前节点与左右子树的最大链条构成一种情况
		return max(root->val + max(left_num, right_num), 0);//路径和可能为负 所以和0取最大值
	};
	dfs(root);
	return ans;
}
```

一般树
#### [2246. 相邻字符不同的最长路径](https://leetcode.cn/problems/longest-path-with-different-adjacent-characters/)
```c++
int longestPath(vector<int>& parent, string s) {
	int n = s.size();
	vector<vector<int>>list(n);
	for (int i = 1; i < n; i++) {//parent[0]为-1表示根节点
		list[parent[i]].emplace_back(i);
	}
	int ans = 0;
	//如果邻居包含父节点 dfs(i,x)其中x是该节点的父节点 遍历邻居的时候判断是父节点 continue 递归入口dfs(0,-1)
	function<int(int)>dfs = [&](int i)->int {
		int maxlist = 0;//当前节点i的最大链长
		for (const int& y : list[i]) {//遍历邻居 这里邻居都是孩子
			int len_y = dfs(y) + 1;//当前子链的最大长度+与当前点的边
			if (s[y] != s[i]) {//相邻节点没有相同字符
				//maxlist是之前的最长链 len_y是当前子链 两条子链构成一个路径
				ans = max(ans, maxlist + len_y);
				maxlist = max(maxlist, len_y);//更新当前节点的最长链
			}
		}
		return maxlist;//向父节点返回的是一条最长的链
	};
	dfs(0);
	return ans + 1;//ans是边的数量 节点数要+1
}
```

树上最大独立集
#### [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)
```c++
int rob(TreeNode* root) {
	//时空复杂度为O(n)
	//dfs 返回pair<选，不选>      
	function<pair<int, int>(TreeNode*)>dfs = [&](TreeNode* root)->pair<int, int> {
		if (root == nullptr)return { 0,0 };
		pair<int, int>left = dfs(root->left);
		pair<int, int>right = dfs(root->right);
		//选：当前节点的值+左孩子不选+右孩子不选  =>一般树：选：当前节点的值+ sum(不选子节点)
		int rob = root->val + left.second + right.second;
		//不选：左孩子选或不选的最大值+右孩子选或不选的最大值 =>一般树：不选：sum(max(选，不选子节点))
		int not_rob = max(left.first, left.second) + max(right.first, right.second);
		return { rob,not_rob };
	};
	pair<int, int>ans = dfs(root);
	return max(ans.first, ans.second);
}
```
[1377. T 秒后青蛙的位置](https://leetcode.cn/problems/frog-position-after-t-seconds/)
```c++
```
树上最小支配集
#### [968. 监控二叉树](https://leetcode.cn/problems/binary-tree-cameras/)
```c++
//A:选 ：在该节点安装摄像头
//B:不选 ：在该节点的父节点安装摄像头  
//C:不选 ：在该节点的左/右孩子安装摄像头  
//A = min(leftA,leftB,leftC) + min(rightA,rightB,rightC) + 1 (1表示当前节点安装摄像头)
//B = min(leftA,rightC) + min(rightA,rightC)  (根据B的定义 左右孩子不能为B)
//C = min(leftA+rightC,leftC+rightA,leftA+rightA) (根据C的定义 左右孩子至少一个为A 且不能为B)
//ans =min(rootA,rootC)      nuullptrA(∞)nullptrB=nullptrC=0
```
## 贪心
### [1029. 两地调度](https://leetcode.cn/problems/two-city-scheduling/)
```c++
int twoCitySchedCost(vector<vector<int>>& costs) {
	//先将N个人去A 剩余N个人去B 
	//如果将其中一个人从A换到B 那么将消耗acost-bcost的钱(可正可负)
	//故直接按(acost-bcost)从小到大排序 前N个去A 剩余去B
	int n = costs.size();
	sort(costs.begin(), costs.end(),
		[](vector<int>& a, vector<int>& b) {
			return a[0] - a[1] < b[0] - b[1]; });
	int ans = 0, i = 0;
	while (i < n) {
		if (i < n / 2) {
			ans += costs[i][0];
			i++;
		}
		else {
			ans += costs[i][1];
			i++;
		}
	}
	return ans;
}
```
### [134. 加油站](https://leetcode.cn/problems/gas-station/)
```c++
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
	int max_ = INT_MIN, max_i; long sum = 0;
	for (int i = gas.size() - 1; ~i; i--) {
		sum += gas[i] - cost[i];//每个位置的后缀和 最终也是总汽油与总花费的差
		if (sum > max_) {
			max_ = sum;
			max_i = i;//记录从哪个位置出发向后能够获得最多的汽油
		}
	}
	return sum >= 0 ? max_i : -1;//如果总花费大于总汽油 肯定没法跑一圈
}
```
### [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

```c++
int maxProfit(vector<int>& prices) {
	int ans = 0;
	for (int i = 0; i < prices.size() - 1; i++) {
		ans += max(0, prices[i + 1] - prices[i]);
	}
	return ans;
}
```

### [1402. 做菜顺序](https://leetcode.cn/problems/reducing-dishes/)

```c++
int maxSatisfaction(vector<int>& satisfaction) {
    int ans = 0;
    int pre = 0;
    int f = 0;
    vector<int>& a = satisfaction;
    class mycompare {
    public:
        bool operator()(int a, int b) {
            return a > b;
        }
    };
    sort(a.begin(), a.end(), mycompare());
    int n = a.size();
    for (int i = 0; i < n; i++) {
        if (pre < 0)break;
        pre += a[i];
        f += pre;
        ans = max(ans, f);
    }
    return ans;
}
```
### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

```c++
int lengthOfLIS(vector<int>& nums) {
	//贪心+二分
	//时间复杂度O(nlogn) 空间复杂度O(n)
	vector<int>f;
	for (const auto& v : nums) {
		auto it = lower_bound(f.begin(), f.end(), v);//在数组中查找元素v
		if (it == f.end()) {//v大于f中所有元素 
			f.emplace_back(v); //插入到数组末尾
		}
		else *it = v;//找到一个大于等于v的数 将其减少至v 增大了数组变长的潜力
	}
	return f.size();
}
```
### [334. 递增的三元子序列](https://leetcode.cn/problems/increasing-triplet-subsequence/)
```c++
bool increasingTriplet(vector<int>& nums) {
	vector<int>v(2, INT_MAX);//用来记录最小元素和次最小元素
	for (const int& num : nums) {
		if (num <= v[0]) v[0] = num;
		else if (num <= v[1])v[1] = num;
		else return true;//说明前面有两个元素小于该元素
	}
	return false;
}
```
### [435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/)
```c++
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
	//按照结束时间从早到晚排序 结束时间相同的 开始时间晚的排在前面
	sort(intervals.begin(), intervals.end(),
		[&](const vector<int>& a, const vector<int>& b) {
			if (a[1] == b[1])return a[0] > b[0];
			return a[1] < b[1]; });
	int low = 0, high = 1;
	while (high < intervals.size()) {
		if (intervals[high][0] >= intervals[low][1]) {
			intervals[++low] = intervals[high];
		}
		high++;
	}
	return high - low - 1;
}
```
## 图

### 深度优先搜索DFS
#### [1466. 重新规划路线](https://leetcode.cn/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/)
```c++
int minReorder(int n, vector<vector<int>>& connections) {
	vector<vector<pair<int, int>>>edges(n);
	//存储与每个节点相关的边pair<int,int>
	//pair.first表示起点 pair.second表示终点
	for (const auto& con : connections) {
		pair<int, int>t = make_pair(con[0], con[1]);
		edges[con[0]].emplace_back(t);
		edges[con[1]].emplace_back(t);
	}
	int ans = 0;
	function<void(int, int)>dfs = [&](int i, int p) {
		for (const auto& ed : edges[i]) {
			if (ed.first == i) {//i是起点 表示从根节点向外的方向 需要修改这条路
				if (ed.second != p && ed.second) {//终点不是父节点 且不是根节点0
					ans++;
					dfs(ed.second, i);
				}
			}
			else {//i是终点 表示朝根节点的方向 不用修改
				if (ed.first != p && ed.first) {//起点不是父节点 且不是根节点0
					dfs(ed.first, i);
				}
			}
		}
	};
	dfs(0, -1);
	return ans;
}
```
### 广度优先搜索BFS

## 贡献法【计算所有子数组的……的和】(正难则反)
