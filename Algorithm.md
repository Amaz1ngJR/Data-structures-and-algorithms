Algorithm

![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/3f60b9b4-4d9a-4d77-8bca-82876d343086)

# 二分算法
二分查找

适用：排好序的数组
```c++
//二分查找 有序数组
template<typename T>
int BinarySearch(vector<T>& nums, T target) {//时间复杂度O(logn)
	//闭区间写法[left,right]
	int left = 0, right = nums.size() - 1, mid;
	if (nums[0] >= target) return 0;
	if (nums[nums.size() - 1] < target) return nums.size();
	while (left <= right) {
		mid = (left + right) / 2;
		if (nums[mid] == target) return mid;//找到返回mid
		if (nums[mid] > target) right = mid - 1;
		if (nums[mid] < target) left = mid + 1;
	}
	return left;//找不到就返回target应该插入的位置

	//开区间写法(low,high)
	int low = -1, high = nums.size(), mid;
	while (low + 1 < high) {
		mid = low + (high - low) / 2;
		if (nums[mid] == target)return mid;
		else if (nums[mid] > target)
			high = mid;
		else
			low = mid;
	}
	return high;//找不到就返回target应该插入的位置
}
```

## 二分查找
pre [852. 山脉数组的峰顶索引](https://leetcode.cn/problems/peak-index-in-a-mountain-array/)
### [162. 寻找峰值](https://leetcode.cn/problems/find-peak-element/)
```c++
int findPeakElement(vector<int>& nums) {
	int low = -1, high = nums.size() - 1, mid;
	while (low + 1 < high) {
		mid = low + (high - low) / 2;
		(nums[mid] > nums[mid + 1] ? high : low) = mid;
	}
	return high;
}
```
do [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

### [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

可以使用153的方法先找到两个有序数组，再二分。仅一次二分的方法如下

二分位置nums[mid]右侧全是蓝色的三种情况：

1、如果nums[n-1]<target<=nums[mid] 同一左段 

2、nums[mid]<nums[n-1]<target    target左段、mid右段

3、target<=nums[mid]<nums[n-1]    同一右段   

```c++
int search(vector<int>& nums, int target) {
        int n = nums.size(), low = -1, high = n - 1, mid;
        auto check = [&](int mid)->bool { // 能否将high左移到mid
            if(nums[mid] > nums.back()) { //mid在第一段
                //只有target也在第一段的时候 且满足正常二分条件
                return target > nums.back() && nums[mid] >= target;
            }
            else { //mid在第二段
                //target在第一段 或者 target在第二段且满足正常二分条件
                return target > nums.back() || nums[mid] >= target;
            }
        };
        while(low + 1 < high) {
            mid = low + (high - low) / 2;
            (check(mid) ? high : low) = mid;
        }
	return nums[high] == target ? high : - 1;
}
```
### [1901. 寻找峰值 II](https://leetcode.cn/problems/find-a-peak-element-ii/)
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
### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)
```c++
int lengthOfLIS(vector<int>& nums) {
	//贪心+二分
	//时间复杂度O(nlogn) 空间复杂度O(n)
	vector<int>f;
	for (const auto& v : nums) {
		auto it = lower_bound(f.begin(), f.end(), v);//在f中查找第一个大于或等于v的位置
		if (it == f.end()) {//f中所有元素都小于v
			f.emplace_back(v); //插入到数组末尾
		}
		else *it = v;//找到一个大于等于v的数 将其减少至v 增大了数组变长的潜力
	}
	return f.size();
}
```
## 二分答案
看到「最大化最小值」或者「最小化最大值」就要想到二分答案 这是一个固定的套路
### *最小化最大值
pre [1283. 使结果不超过阈值的最小除数](https://leetcode.cn/problems/find-the-smallest-divisor-given-a-threshold/)、
[2187. 完成旅途的最少时间](https://leetcode.cn/problems/minimum-time-to-complete-trips/)、
[2064. 分配给商店的最多商品的最小值](https://leetcode.cn/problems/minimized-maximum-of-products-distributed-to-any-store/)、
[1760. 袋子里最少数目的球](https://leetcode.cn/problems/minimum-limit-of-balls-in-a-bag/)、
[1011. 在 D 天内送达包裹的能力](https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/)
#### [410. 分割数组的最大值](https://leetcode.cn/problems/split-array-largest-sum/)
```c++
int splitArray(vector<int>& nums, int k) {
	auto check = [&](int& mx)->bool {//检查子数组的最大和为mx的情况下能不能划分小于等于k段子数组
		int cnt = 1, s = 0;//cnt为当前划分的段数 s为当前段(子数组)的和
		for (const int& x : nums) {
			if (s + x <= mx) s += x;
			else {// 新划分一段
				if (cnt++ == k)return false;//cnt为k则已经划分k段了无法新划分了
				s = x;
			}
		}
		return true;
	};
	//开区间写法 
	int high = accumulate(nums.begin(), nums.end(), 0);//右边界设为S(nums)肯定满足 k最小为1
	//左边界设为max(nums)-1肯定不满足 每个子数组和都小于S/k也就是(S-1)/k也肯定不满足 k最大为数组长度时也不满足
	int low = max(*max_element(nums.begin(), nums.end()) - 1, (high - 1) / k);
	while (low + 1 < high) {//mid越大分段数cnt越小 题意是得到最小的mid且cnt不超过k
		int mid = low + (high - low) / 2;
		(check(mid) ? high : low) = mid;//要加上括号
	}
	return high;
}
```
#### [2226. 每个小孩最多能分到多少糖果](https://leetcode.cn/problems/maximum-candies-allocated-to-k-children/)
```c++
int maximumCandies(vector<int>& candies, long long k) {
	int low = 0, high = 1 + (*max_element(begin(candies), end(candies))), mid;
	auto check = [&](int& mid)->bool {
		long long sum = 0;
		for (int i = 0; i < candies.size(); i++) 
			sum += candies[i] / mid;
		return sum >= k;
	};
	while (low + 1 < high) {//开区间写法
		mid = low + (high - low) / 2;
		(check(mid) ? low : high) = mid;
	}
	return low;
}
```
```c++
int maximumCandies(vector<int>& candies, long long k) {
	int low = 1, high = *max_element(begin(candies), end(candies)), mid;
	auto check = [&](int& mid)->bool {
		long long sum = 0;
		for (int i = 0; i < candies.size(); i++) 
			sum += candies[i] / mid;
		return sum >= k;
	};
	while (low <= high) {//闭区间写法
		mid = low + (high - low) / 2;
		if (check(mid)) low = mid + 1;
		else high = mid - 1;
	}
	return low - 1;
}
```
#### [2439. 最小化数组中的最大值](https://leetcode.cn/problems/minimize-maximum-of-array/)
```c++
int minimizeArrayValue(vector<int>& nums) {
	int n = nums.size(), low = 0, high = *max_element(begin(nums), end(nums)), mid;
	auto check = [&](int& mid)->bool {
		long long extra = 0;
		for (int i = n - 1; i > 0; i--)
			extra = max((long long)nums[i] - mid + extra, (long long)0);
		return nums[0] + extra <= mid;
	};
	while (low + 1 < high) {
		mid = low + (high - low) / 2;
		(check(mid) ? high : low) = mid;
	}
	return high;
}
```
#### [2560. 打家劫舍 IV](https://leetcode.cn/problems/house-robber-iv/)
```c++
int minCapability(vector<int>& nums, int k) {
	int n = nums.size(), low = 0, high = *max_element(begin(nums), end(nums)) + 1, mid;
	auto check = [&](int& mid)->bool {
		int cnt = 0, last = -2;
		for (int i = 0; i < n; i++) {
			if (nums[i] <= mid && i - 1 != last) {
				last = i;
				if (++cnt == k)return true;
			}
		}
		return false;
	};
	while (low + 1 < high) {
		mid = low + (high - low) / 2;
		(check(mid) ? high : low) = mid;
	}
	return high;
}
```
### *最大化最小值
pre [1870. 准时到达的列车最小时速](https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/)、
#### [1552. 两球之间的磁力](https://leetcode.cn/problems/magnetic-force-between-two-balls/)
```c++
int maxDistance(vector<int>& position, int m) {
	sort(position.begin(), position.end());
	int n = position.size(), low = 0, high = position[n - 1] - position[0] + 1, mid;
	auto check = [&](int& mid)->bool {
		int cnt = 1, last = 0;
		for (int i = 1; i < n; i++) {
			if (position[i] - position[last] >= mid) {
				last = i;
				if (++cnt == m)return true;
			}
		}
		return false;
	};
	while (low + 1 < high) {
		mid = low + (high - low) / 2;
		(check(mid) ? low : high) = mid;
	}
	return low;
}
```
#### [2861. 最大合金数](https://leetcode.cn/problems/maximum-number-of-alloys/)
```c++
int maxNumberOfAlloys(int n, int k, int budget,
	vector<vector<int>>& composition,
	vector<int>& stock, vector<int>& cost) {
	int low = -1, mid, high = 1e9;//数据刁钻
	auto check = [&](int& mid)->bool {//合金个数
		int min_cost = INT_MAX; long long cur_cost, need;//数据刁钻
		for (const auto& com : composition) {//枚举所有机器
			cur_cost = 0;
			for (int i = 0; i < n; i++) {
				need = (long long)mid * com[i] - stock[i];
				if (need > 0)cur_cost += need * cost[i];
			}
			min_cost = (cur_cost < min_cost) ? cur_cost : min_cost;
		}
		return min_cost <= budget;
	};
	while (low + 1 < high) {
		mid = low + (high - low) / 2;
		(check(mid) ? low : high) = mid;
	}
	return low;
}
```
## 矩形二分(双指针)
### [搜索二维矩阵 II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)
```c++
bool searchMatrix(vector<vector<int>>& matrix, int target) {
	int m = matrix.size(), n = matrix[0].size();
	int i = 0, j = n - 1;
	while(i < m && j >= 0) {
	    if(matrix[i][j] == target) return true;
	    else if(matrix[i][j] > target) --j;
	    else ++i;
	}
	return false;
}
```
### [378. 有序矩阵中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-sorted-matrix/)
```c++
int kthSmallest(vector<vector<int>>& matrix, int k) {
	int m = matrix.size(), n = matrix[0].size();
	int low = matrix[0][0] - 1, high = matrix[m - 1][n -1];
	auto check = [&](int mid)->bool {
	    int cnt = 0;
	    int i = 0, j = n - 1;
	    while(i < m && j >= 0 && cnt < k) {
		if(matrix[i][j] <= mid) {
		    //cnt = max(cnt, i * n + j + 1);
		    cnt += j + 1;
		    ++i;
		}
		else --j;
	    }
	    return cnt >= k;
	};
	while(low + 1 < high) {
	    int mid = low + (high - low) / 2;
	    (check(mid) ? high : low) = mid;
	}
	return high;
}
```
# 前后缀与差分数组

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
vector<vector<int>>sum(m + 1, vector<int>(n + 1, 0));//二维前缀和 下标是从1开始的！
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

## *差分
### [1094. 拼车](https://leetcode.cn/problems/car-pooling/)
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
### [2132. 用邮票贴满网格图](https://leetcode.cn/problems/stamping-the-grid/) 二维差分
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
## *前后缀
### [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)
```c++
int maxProfit(vector<int>& prices) {
	int ans = 0, n = prices.size(), maxsuf = 0;
	vector<int>suf(n);
	for (int i = n - 1; ~i; --i) {
		suf[i] = maxsuf;
		maxsuf = max(maxsuf, prices[i]);
	}
	for (int i = 0; i < n; ++i) {
		ans = max(ans, suf[i] - prices[i]);
	}
	return ans;
}
```
### [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)
```c++
vector<int> productExceptSelf(vector<int>& nums) {
	int n = nums.size(), suf = 1;
	vector<int> ans(n, 1);
	for (int i = 1; i < n; ++i) {
		ans[i] = nums[i - 1] * ans[i - 1];//ans是前缀积
	}
	for (int i = n - 1; ~i; --i) {
		ans[i] *= suf;
		suf *= nums[i];//suf是后缀积
	}
	return ans;
}
```
### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

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
### [560. 和为 K 的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/) 前缀和+哈希表
```c++
int subarraySum(vector<int>& nums, int k) {
	int ans = 0, sum = 0;
	unordered_map<int, int>cnt;
	cnt[0] = 1;//特殊情况 前缀和为0是存在的
	for(const int& num: nums) {
	    sum += num;//前缀和
	    ans += cnt[sum - k];//找到前缀和为sum - k 的个数
	    ++cnt[sum];
	}
	return ans;
}

```
### [2602. 使数组元素全部相等的最少操作次数](https://leetcode.cn/problems/minimum-operations-to-make-all-array-elements-equal/) 前缀和+二分查找
```c++
vector<long long> minOperations(vector<int>& nums, vector<int>& queries) {
	int n = nums.size(), m = queries.size();
	vector<long long>ans(m), pre_sum(n + 1);
	ranges::sort(nums);
	for (int i = 0; i < n; ++i)
		pre_sum[i + 1] = pre_sum[i] + nums[i];
	for (int i = 0; i < m; ++i) {
		int q = queries[i];
		long long len = ranges::lower_bound(nums, q) - nums.begin();
		long long left = q * len - pre_sum[len];
		long long right = pre_sum[n] - pre_sum[len] - q * (n - len);
		ans[i] = left + right;
	}
	return ans;
}
```
### [221. 最大正方形](https://leetcode.cn/problems/maximal-square/) 二维前缀和+二分答案
```c++
int maximalSquare(vector<vector<char>>& matrix) {
	int m = matrix.size(), n = matrix[0].size(), low = -1, high = min(m, n) + 1;
	vector<vector<int>>sum(m + 1, vector<int>(n + 1, 0));//二维前缀和 下标从1开始
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			sum[i + 1][j + 1] = sum[i + 1][j] + sum[i][j + 1] - sum[i][j] + (matrix[i][j] == '1');
	auto check = [&](int mid) {
		for (int x = 0; x + mid - 1 < m; ++x)
			for (int y = 0; y + mid - 1 < n; ++y)
				if (sum[x + mid][y + mid]
					- sum[x + mid][y]
					- sum[x][y + mid]
					+ sum[x][y] == mid * mid)
					return true;
		return false;
	};
	while (low + 1 < high) {
		int mid = low + (high - low) / 2;
		(check(mid) ? low : high) = mid;
	}
	return low * low;
}
```
### [437. 路径总和 III](https://leetcode.cn/problems/path-sum-iii/)
```c++
int pathSum(TreeNode* root, int targetSum) {
	int ans = 0; long long sum = 0;
	vector<int>path;
	path.push_back(0);//存在一个0 根节点就是target的话也能正确
	function<void(TreeNode*)>dfs = [&](TreeNode* root) {
		if (!root) return;
		sum += root->val;//选了
		if (int a = count(path.begin(), path.end(), sum - targetSum))
			ans += a;//以当前结点作为路径结尾的满足条件的路径个数
		path.push_back(sum);
		dfs(root->left);
		dfs(root->right);
		//恢复现场
		sum -= root->val;
		path.pop_back();
	};
	dfs(root);
	return ans;
}
```
# 双指针
### [160. 相交链表](https://leetcode.cn/problems/intersection-of-two-linked-lists/)
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
### [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)
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
## *相向双指针

### [167. 两数之和 II - 输入有序数组](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)

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

### [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)
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
### [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)
```c++
int maxArea(vector<int>& height) {
	int low = 0, high = height.size() - 1, ans = 0;
	while (low < high) {
		ans = max(ans, (high - low) * min(height[low], height[high]));
		if (height[low] < height[high])++low;
		else --high;
	}
	return ans;
}
```
## *同向双指针
#### [88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)
```c++
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
//逆向的同向双指针 
//指针设置为从后向前遍历，每次取两者之中的较大者放进 nums 1的最后面。
	int tail = m + n - 1, p1 = m - 1, p2 = n - 1;
	while(p1 >= 0 || p2 >= 0) {
	    if(p1 < 0) {
		nums1[tail--] = nums2[p2--];
	    }
	    else if(p2 < 0) {
		nums1[tail--] = nums1[p1--];
	    }
	    else {
		if(nums1[p1] > nums2[p2]) {
		    nums1[tail--] = nums1[p1--];
		}
		else {
		    nums1[tail--] = nums2[p2--];
		}
	    }
	}
}
```
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
### 定长滑动窗口
pre [1343. 大小为 K 且平均值大于等于阈值的子数组数目](https://leetcode.cn/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/)、
[2090. 半径为 k 的子数组平均值](https://leetcode.cn/problems/k-radius-subarray-averages/)、
[2379. 得到 K 个黑块的最少涂色次数](https://leetcode.cn/problems/minimum-recolors-to-get-k-consecutive-black-blocks/)、
[1052. 爱生气的书店老板](https://leetcode.cn/problems/grumpy-bookstore-owner/)、
[2653. 滑动子数组的美丽值](https://leetcode.cn/problems/sliding-subarray-beauty/)、
#### [2841. 几乎唯一子数组的最大和](https://leetcode.cn/problems/maximum-sum-of-almost-unique-subarray/)
```c++
long long maxSum(vector<int>& nums, int m, int k) {
	int n = nums.size();
	long long sum = 0, ans = 0;
	unordered_map<int, int>um;
	for (int i = 0; i < n; i++) {
		sum += nums[i];
		um[nums[i]]++;
		if (i >= k - 1) {
			if (um.size() >= m)ans = max(ans, sum);
			sum -= nums[i - k + 1];
			if (--um[nums[i - k + 1]] == 0)//及时删除值为0的键
				um.erase(nums[i - k + 1]);
		}
	}
	return ans;
}
```
#### [2134. 最少交换次数来组合所有的 1 II](https://leetcode.cn/problems/minimum-swaps-to-group-all-1s-together-ii/)
```c++
int minSwaps(vector<int>& nums) {
	int n = nums.size(), k = 0, sum = 0, ans = INT_MAX;
	for (const int& num : nums)k += num;
	for (int i = 0; i < 2 * n; i++) {
		sum += nums[i % n];
		if (i >= k - 1) {
			ans = min(ans, k - sum);
			sum -= nums[(i - k + 1) % n];
		}
	}
	return ans;
}
```
#### [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)
```c++
vector<int> findAnagrams(string s, string p) {
	vector<int>ans;
	vector<int>cnt(26), window(26);
	for (const auto& ch : p)++cnt[ch - 'a'];
	int n = s.size(), low = 0, high = 0;
	while (high < n) {
		++window[s[high] - 'a'];
		while (window[s[high] - 'a'] > cnt[s[high] - 'a']) 
			--window[s[low++] - 'a'];
		if (window == cnt)ans.emplace_back(low);
		++high;
	}
	return ans;
}
```
#### [2156. 查找给定哈希值的子串](https://leetcode.cn/problems/find-substring-with-given-hash-value/)
```c++
string subStrHash(string s, int power, int modulo, int k, int hashValue) {
	int ans = -1, n = s.size();
	long long hash = 0, mult = 1;
	for (int i = n - 1; i > n - k; i--) {
		hash = power * (hash + s[i] - 96) % modulo;
		mult = mult * power % modulo;//power^(k-1) mod modulo
	}
	for (int i = n - k; ~i; i--) {//窗口开始满
		hash = (hash + s[i] - 96) % modulo;
		if (hash == hashValue)ans = i;
		hash = power * (hash + modulo - mult * (s[i + k - 1] - 96) % modulo) % modulo;
	}
	return s.substr(ans, k);
}
```
### 不定长滑动窗口

#### [1004. 最大连续1的个数 III](https://leetcode.cn/problems/max-consecutive-ones-iii/)
```c++
int longestOnes(vector<int>& nums, int k) {
	int low = 0, high = 0, ans = 0;
	while (high != nums.size()) {
		if (!nums[high]) k--;//遇到0就填 直到不满足条件
		while (k < 0) {// 条件不满足移动左指针
			if (!nums[low]) k++;
			low++;
		}
		ans = max(ans, high - low + 1);//时刻记录最大值
		high++;
	}
	return ans;
}
```

[209. 长度最小的子数组](https://leetcode.cn/problems/minimum-size-subarray-sum/)、
[3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)、
[1493. 删掉一个元素以后全为 1 的最长子数组](https://leetcode.cn/problems/longest-subarray-of-1s-after-deleting-one-element/)、
[2730. 找到最长的半重复子字符串](https://leetcode.cn/problems/find-the-longest-semi-repetitive-substring/)、
[1695. 删除子数组的最大得分](https://leetcode.cn/problems/maximum-erasure-value/)、
[2958. 最多 K 个重复元素的最长子数组](https://leetcode.cn/problems/length-of-longest-subarray-with-at-most-k-frequency/)、
[2024. 考试的最大困扰度](https://leetcode.cn/problems/maximize-the-confusion-of-an-exam/)、
[2401. 最长优雅子数组](https://leetcode.cn/problems/longest-nice-subarray/)、
[2302. 统计得分小于 K 的子数组数目](https://leetcode.cn/problems/count-subarrays-with-score-less-than-k/)、
[1658. 将 x 减到 0 的最小操作数](https://leetcode.cn/problems/minimum-operations-to-reduce-x-to-zero/)、
[1838. 最高频元素的频数](https://leetcode.cn/problems/frequency-of-the-most-frequent-element/)、

pre [2799. 统计完全子数组的数目](https://leetcode.cn/problems/count-complete-subarrays-in-an-array/)
#### [1358. 包含所有三种字符的子字符串数目](https://leetcode.cn/problems/number-of-substrings-containing-all-three-characters/)
```c++
int numberOfSubstrings(string s) {
	vector<int>cnt(3, 0);
	int n = s.size(), low = 0, high = 0, ans = 0;
	while (high < n) {
		cnt[s[high] - 'a']++;
		while (cnt[0] && cnt[1] && cnt[2]) {
			ans += n - high;//[low, high]到[low, n-1]都满足
			cnt[s[low] - 'a']--;
			low++;
		}
		high++;
	}
	return ans;
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
#### [904. 水果成篮](https://leetcode.cn/problems/fruit-into-baskets/)
```c++
int totalFruit(vector<int>& fruits) {
	int n = fruits.size(), low = 0, high = 0, ans = 0;
	unordered_map<int, int>m;
	while (high < n) {
		m[fruits[high]]++;
		while (m.size() > 2) {
			if (--m[fruits[low]] == 0) {
				m.erase(fruits[low++]);
				break;
			}
			low++;
		}
		ans = max(ans, high - low + 1);
		high++;
	}
	return ans;
}
```
#### [1438. 绝对差不超过限制的最长连续子数组](https://leetcode.cn/problems/longest-continuous-subarray-with-absolute-diff-less-than-or-equal-to-limit/)
do [2762. 不间断子数组](https://leetcode.cn/problems/continuous-subarrays/)
```c++
int longestSubarray(vector<int>& nums, int limit) {
	multiset<int>ms;
	int n = nums.size(), low = 0, high = 0, ans = 0;
	while (high < n) {
		ms.emplace(nums[high]);
		while (*ms.rbegin() - *ms.begin() > limit) {
			ms.erase(ms.find(nums[low++]));
		}
		ans = max(ans, high - low + 1);
		high++;
	}
	return ans;
}
```

#### [2962. 统计最大元素出现至少 K 次的子数组](https://leetcode.cn/problems/count-subarrays-where-max-element-appears-at-least-k-times/)
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

#### [1234. 替换子串得到平衡字符串](https://leetcode.cn/problems/replace-the-substring-for-balanced-string/)
```c++
int balancedString(string s) {
	int n = s.size(), t = n / 4;
	unordered_map<char, int>m = { {'Q',0},{'W',1},{'E',2},{'R',3} };
	vector<int>cnt(4, 0);
	for (const char& ch : s)cnt[m[ch]]++;
	if (cnt[0] == t && cnt[1] == t && cnt[2] == t && cnt[3] == t)return 0;
	int ans = n, low = 0, high = 0;
	while (high < n) {
		cnt[m[s[high]]]--;
		while (cnt[0] <= t && cnt[1] <= t && cnt[2] <= t && cnt[3] <= t) {
			ans = min(ans, high - low + 1);
			cnt[m[s[low]]]++;
			low++;
		}
		high++;
	}
	return ans;
}
```
#### [1574. 删除最短的子数组使剩余数组有序](https://leetcode.cn/problems/shortest-subarray-to-be-removed-to-make-array-sorted/)
```c++
int findLengthOfShortestSubarray(vector<int>& arr) {
	int n = arr.size(), low = 0, high = n - 1;
	while (high && arr[high - 1] <= arr[high])
		high--;
	int ans = high;//删除[0,high-1]
	while (high) {
		while (high == n || arr[low] <= arr[high]) {
			ans = min(ans, high - low - 1);//删除[low+1,high-1]
			if (arr[low] > arr[low + 1])return ans;
			low++;
		}
		high++;
	}
	return ans;
}
```
#### [2516. 每种字符至少取 K 个](https://leetcode.cn/problems/take-k-of-each-character-from-left-and-right/)
```c++
int takeCharacters(string s, int k) {
	vector<int>tar(3, 0), cnt(3, 0);
	for (const char& ch : s)tar[ch - 'a']++;
	for (int& ta : tar) {
		ta -= k;
		if (ta < 0)return -1;
	}
	int n = s.size(), ans = INT_MAX, low = 0, high = 0, i;
	while (high < n) {
		i = s[high] - 'a';
		cnt[i]++;
		while (cnt[i] > tar[i]) {
			cnt[s[low] - 'a']--;
			low++;
		}
		ans = min(ans, n - high + low - 1);
		high++;
	}
	return ans;
}
```
#### [2537. 统计好子数组的数目](https://leetcode.cn/problems/count-the-number-of-good-subarrays/)
```c++
long long countGood(vector<int>& nums, int k) {
	int n = nums.size(), low = 0, high = 0, sum = 0;
	long long ans = 0;
	unordered_map<int, int>cnt;
	while (high < n) {//枚举子数组右端点high
		sum += cnt[nums[high]]++;//新增的满足的对数
		//如果源区间[low,high]恰好满足那么除了源区间 向左[0,high]-[low-1,high]还有low个满足
		//故随着右端点high++都有low个区间满足(固定左端点计算)
		ans += low;
		while (sum >= k) {//满足条件
			ans++;//[low,high]恰好满足条件的源区间本身
			sum -= --cnt[nums[low++]];
		}
		high++;
	}
	return ans;
}
```
### 多指针滑动窗口
#### [930. 和相同的二元子数组](https://leetcode.cn/problems/binary-subarrays-with-sum/)
do [1248. 统计「优美子数组」](https://leetcode.cn/problems/count-number-of-nice-subarrays/)
```c++
int numSubarraysWithSum(vector<int>& nums, int goal) {
	int n = nums.size(), low1 = 0, low2 = 0, high = 0, sum1 = 0, sum2 = 0, ans = 0;
	while (high < n) {
		sum1 += nums[high];
		while (low1 <= high && sum1 > goal)//[low1,high]恰好满足
			sum1 -= nums[low1++];
		sum2 += nums[high];
		while (low2 <= high && sum2 >= goal)//[low2,high]恰好不满足
			sum2 -= nums[low2++];
		ans += low2 - low1;//以high为窗口右端点 左端点落在[low1,low2)之间都满足
		high++;
	}
	return ans;
}
```
#### [2563. 统计公平数对的数目](https://leetcode.cn/problems/count-the-number-of-fair-pairs/)
```c++
long long countFairPairs(vector<int>& nums, int lower, int upper) {
	sort(nums.begin(), nums.end());
	int n = nums.size(); long long ans = 0;
	for (int j = 0; j < n; j++) {//枚举j => lower - nums[j] <= nums[i] <=upper - nums[j] [left,right]
		auto low = lower_bound(nums.begin(), nums.begin() + j, lower - nums[j]);//[0,j]中第一个大于或等于left的
		auto high = upper_bound(nums.begin(), nums.begin() + j, upper - nums[j]);//[0,j]中第一个大于right的 即right+1
		ans += high - low;
	}
	return ans;
}
```
#### [1712. 将数组分成三个子数组的方案数](https://leetcode.cn/problems/ways-to-split-array-into-three-subarrays/)
```c++
int waysToSplit(vector<int>& nums) {
	int n = nums.size(), ans = 0, low1, low2, high = 2, mod = 1e9 + 7;//[0,low)、[low,high)、[high,n)
	vector<long>pre(n + 1);//pre[l]<=pre[h]-pre[l]<=pre[n]-pre[h] => 2pre[h]-pre[n] <= pre[l] <= pre[h]/2 
	partial_sum(nums.begin(), nums.end(), pre.begin() + 1);//sum[low,high)=pre[high]-pre[low]
	while (high < n && 3 * pre[high] <= 2 * pre[n]) {//pre[h]>=4pre[h]-2pre[n] =>2*pre[n]>=3pre[high]
		low1 = lower_bound(pre.begin() + 1, pre.begin() + high, 2 * pre[high] - pre[n]) - pre.begin();
		low2 = upper_bound(pre.begin() + low1, pre.begin() + high, pre[high] / 2) - pre.begin();
		ans = (ans + low2 - low1) % mod;
		high++;
	}
	return ans;
}
```
#### [2444. 统计定界子数组的数目](https://leetcode.cn/problems/count-subarrays-with-fixed-bounds/)
```c++
long long countSubarrays(vector<int>& nums, int minK, int maxK) {
	int n = nums.size(), min_i = -1, max_i = -1, low = -1, high = 0;
	long long ans = 0;
	while (high < n) {
		int x = nums[high];
		if (x == minK)min_i = high;
		if (x == maxK)max_i = high;
		if (x<minK || x>maxK)low = high;//子数组不能包含low
		ans += max(min(max_i, min_i) - low, 0);
		high++;
	}
	return ans;
}
```
## *快慢指针

### [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

判断一个链表是否是有环
```c++
bool hasCycle(ListNode* head) {
	ListNode* fast = head, * slow = head;
	while (fast) {
		fast = fast->next;
		if (!fast)return false;
		fast = fast->next;
		slow = slow->next;
		if (slow == fast)return true;
	}
	return false;
}
```

### [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)

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
ListNode* detectCycle(ListNode* head) {
	ListNode* slow = head, * fast = head;
	while (fast) {
		fast = fast->next;
		if (!fast) return nullptr;
		fast = fast->next;
		slow = slow->next;
		if (slow == fast) {
			while (head != slow) {
				slow = slow->next;
				head = head->next;
			}
			return head;
		}
	}
	return nullptr;
}
```
### [287. 寻找重复数](https://leetcode.cn/problems/find-the-duplicate-number/)
```c++
int findDuplicate(vector<int>& nums) {
	int slow = 0, fast = 0;
	do {
	    slow = nums[slow];
	    fast = nums[nums[fast]];
	} while(slow != fast);
	slow = 0;
	while(slow != fast){
	    slow = nums[slow];
	    fast = nums[fast];
	}
	return slow;
}
```
### [202. 快乐数](https://leetcode.cn/problems/happy-number/)
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
## *前后指针
### [283. 移动零](https://leetcode.cn/problems/move-zeroes/)
```c++
void moveZeroes(vector<int>& nums) {
	int n = nums.size(), back = 0, front = 0;
	while (front < n) {
		if (nums[front]) {
			nums[back] = nums[front];
			back++;
		}
		front++;
	}
	while (back < n) 
		nums[back++] = 0;
}
```
### [2181. 合并零之间的节点](https://leetcode.cn/problems/merge-nodes-in-between-zeros/)
```c++
ListNode* mergeNodes(ListNode* head) {
	ListNode * dummy = new ListNode();
	dummy->next = head;
	ListNode* front = head->next, * back = dummy;
	int sum = 0;
	while(front){
	    if(front->val == 0) {
		back = back->next;
		back->val = sum;
		sum = 0;
	    }
	    else {
		sum += front->val;
	    }
	    front = front->next;
	}
	back->next = nullptr;
	return head;
}
```
### [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

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

### [83. 删除排序链表中的重复元素](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/)

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
	if (!head)return head;
	ListNode* dummy = new ListNode(0, head), * pre = dummy, * back, * front;
	back = dummy->next, front = dummy->next->next;
	while (back && front) {
		int val = back->val;
		if (front->val == val) {
			while (back && back->val == val) 
				back = back->next;
			pre->next = back;
			if (back)front = back->next;
		}
		else {
			pre->next = back;
			pre = back;
			back = front;
			front = front->next;
		}
	}
	return dummy->next;
}
```

# 回溯

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

## *子集型

### [78. 子集](https://leetcode.cn/problems/subsets/)

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

## *分割型

### [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

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

### [131. 分割回文串](https://leetcode.cn/problems/palindrome-partitioning/)

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

## *组合型
组合数学公式 C(n,k)= n!/(k! * (n-k)!)  组合无序 因此要除以 k!

剪枝技巧： 逆序枚举

```
设path长为m
那么还需选d=k-m个数
设当前要从[1,i]这i个数中选数
如果i<d最后必然没法选够k个数 不需要继续递归
```

### [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

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

### [39. 组合总和](https://leetcode.cn/problems/combination-sum/)

从一个数组中选取一些可重复数使得其和为target，返回所有组合

```c++
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>>ans;
        vector<int> path;
        int n = candidates.size();
        auto& c = candidates;
        function<void(int, int)>dfs = [&](int i, int res) {
            if(res == 0) {
                ans.emplace_back(path);
                return;
            }
            if(i == n || res < 0) return;
            //不选
            dfs(i + 1, res);
            //选
            path.emplace_back(c[i]);
            dfs(i, res - c[i]);//可以重复选，还从i开始dfs
            path.pop_back();//恢复现场
        };
        dfs(0, target);
        return ans;
}
```

### [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)

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
### [22. 括号生成](https://leetcode.cn/problems/generate-parentheses/)
```c++
vector<string> generateParenthesis(int n) {
        vector<string>ans;
        int m = 2 * n;
	string path(m, 0);
        function<void(int, int)> dfs = [&](int i, int open) {//open表示左括号个数
            if(i == m) {
                ans.emplace_back(path);
                return;
            }
            if(open < n) {//可以填左括号
                path[i] = '(';
                dfs(i + 1, open + 1);
            }
            if(i - open < open) {//右括号少于左括号 可以填右括号
                path[i] = ')';
                dfs(i + 1, open);
            }
        };
        dfs(0, 0);
	return ans;	
}
```
## *排列型
排列数学公式 A(n,k)= n!/(n-k)!

不同于组合型，(i,j)!=(j,i)
![image](https://github.com/Amaz1ngJR/Data-structures-and-algorithms/assets/83129567/51adcf91-e69c-466d-839f-1fb280e7734d)

时间复杂度为所有叶子结点到根节点的路径和O(n*n!)

### [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

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

### [51. N 皇后](https://leetcode.cn/problems/n-queens/)

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
## *DFS
### [332. 重新安排行程](https://leetcode.cn/problems/reconstruct-itinerary/)
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
### [37. 解数独](https://leetcode.cn/problems/sudoku-solver/)
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
# 动态规划

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
## *打家劫舍问题

源问题[198. 打家劫舍](https://leetcode.cn/problems/house-robber/)

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
**环形房子的打家劫舍** [213. 打家劫舍 II](https://leetcode.cn/problems/house-robber-ii/)

只要对[0,n-1)和[1,n) 两个区间求两次 取最大即可

**房子间隔不定的打家劫舍** [2140. 解决智力问题](https://leetcode.cn/problems/solving-questions-with-brainpower/description/)
```c++
long long mostPoints(vector<vector<int>>& questions) {
	int n = questions.size();
	vector<long long>memo(n, -1);
	auto dfs = [&](this auto&& dfs, int i)->long long {
	    if(i >= n) return 0;
	    if(memo[i] != -1) return memo[i];
	    memo[i] = (long long)max(questions[i][0] + dfs(i + questions[i][1] + 1), dfs(i + 1));
	    return memo[i];
	};
	return dfs(0);
}
```
**值域打家劫舍** [740. 删除并获得点数](https://leetcode.cn/problems/delete-and-earn/)
```c++
int deleteAndEarn(vector<int>& nums) {
	int mx = ranges::max(nums);
	vector<int> a(mx + 1);
	for (int x : nums) {
	    a[x] += x; // 统计等于 x 的元素之和
	}
	//降为打家劫舍
	int f0 = 0, f1 = 0;
	for (int x : a) {
	    int new_f = max(f1, f0 + x);
	    f0 = f1;
	    f1 = new_f;
	}
	return f1;
	// vector<int>dp(mx + 3);
	// for(int i = 0; i < mx + 1; ++i) {
	//     dp[i + 2] = max(dp[i + 1], dp[i] + a[i]);
	// }
	// return dp[mx + 2];
}
```
## *0/1背包

```
0/1背包：有n个物品  第i个物品的体积为w[i]  价值为v[i]
每个物品至多选一个  求体积和不超过capacity时的最大价值和
dfs(i,c)=max(dfs(i-1,c),dfs(i-1,c-w[i])+v[i]) //背包剩余容量为c时 从前i个物体中能选出的最大价值和
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

### [494. 目标和](https://leetcode.cn/problems/target-sum/)
```c++
//记忆化搜索 dfs(i,c) = dfs(i-1,c) + dfs(i-1,c-w[i])
int findTargetSumWays(vector<int>& nums, int target) {
	//所有前面为+号的数之和为p 所有数之和为s 所有前面为-号的数之和为s-p
	//-(s-p) + p = target => p = (s+t)/2 => s+t为非负偶数
	for (const int& num : nums)target += num;
	if (target < 0 || target & 1)return 0;
	target /= 2;
	int n = nums.size();
	vector<vector<int>>memo(n, vector<int>(target + 1, -1));
	function<int(int, int)>dfs = [&](int i, int c)->int {
		if (i < 0)return c == 0 ? 1 : 0;
		if (memo[i][c] != -1)return memo[i][c];
		if (c < nums[i]) {//当前物体体积大于剩余容量 只能不选
			memo[i][c] = dfs(i - 1, c);
		}
		else memo[i][c] = dfs(i - 1, c) + dfs(i - 1, c - nums[i]);
		return memo[i][c];
	};
	return dfs(n - 1, target);
}
```

```c++
//递推f[i][c]=f[i-1][c]+f[i-1][c-w[i]] => f[i+1][c]=f[i][c]+f[i][c-w[i]]
//数组初始值为边界条件
vector<vector<int>> f(n + 1, vector<int>(target + 1, 0));
f[0][0] = 1;//边界条件作为初始值
for (int i = 0; i < n; i++) {
	for (int c = 0; c <= target; c++) {
		if (c < nums[i])f[i + 1][c] = f[i][c];//只能不选
		else f[i + 1][c] = f[i][c] + f[i][c - nums[i]];
	}
}
return f[n][target];
```

```c++
//滚动数组递推
vector<vector<int>> f(2, vector<int>(target + 1, 0));
f[0][0] = 1;//边界条件作为初始值
for (int i = 0; i < n; i++) {
	for (int j = 0; j < target + 1; j++) {
		if (j < nums[i]) f[(i + 1) % 2][j] = f[i % 2][j];
		else f[(i + 1) % 2][j] = f[i % 2][j] + f[i % 2][j - nums[i]];
	}
}
return f[n % 2][target];
```

```c++
 //一个数组递推
vector<int>dp(target + 1, 0);//dp[i]表示装满容量为i的方案个数
dp[0] = 1;//边界条件 装满一个空背包 是合理的
for (int i = 0; i < n; i++) {
	//如果当前i的值小于容器 就可以装
	for (int c = target; c >= nums[i]; c--) {
		dp[c] += dp[c - nums[i]];
	}
}
return dp[target];
```

```
常见变形：
1、至多装capacity 求方案数/最大价值和
2、恰好装capacity 求方案数/最大/最小价值和
3、至少装capacity 求方案数/最小价值和
```

判断是否能恰好装满背包
### [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)
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
	vector<int>dp(target + 1, 0);//dp[i]能否装满容量为i的背包
	dp[0] = 1;//边界条件 装满一个空背包 是合理的
	for (int i = 0; i < n; i++) {
		for (int c = target; c >= nums[i]; c--) {
			dp[c] |= dp[c - nums[i]];
		}
	}
	return dp[target];
}
```
### [1049. 最后一块石头的重量 II](https://leetcode.cn/problems/last-stone-weight-ii/)
```c++
int lastStoneWeightII(vector<int>& stones) {
	int total = accumulate(begin(stones), end(stones), 0);
	int half = total / 2, n = stones.size();
	vector<int>dp(total + 1, 0);//dp[i]表示容量为i的背包可以装的最大重量
	for (int i = 0; i < n; i++) {
		for (int c = half; c >= stones[i]; c--) {
			dp[c] = max(dp[c], dp[c - stones[i]] + stones[i]);
		}
	}
	return (total - dp[half]) - dp[half];//重的一堆石头减去轻的一堆石头
}
```
### [474. 一和零](https://leetcode.cn/problems/ones-and-zeroes/) 二维0/1背包
```c++
int findMaxForm(vector<string>& strs, int m, int n) {
	int len = strs.size(), cnt0, cnt1;
	vector<vector<int>>dp(m + 1, vector<int>(n + 1, 0));
	for (int i = 0; i < len; i++) {//枚举字符串
		cnt0 = cnt1 = 0;
		for (const char& s : strs[i]) {
			if (s == '0')cnt0++;
			else cnt1++;
		}
		for (int c0 = m; cnt0 <= c0; c0--) {//分别枚举两个背包
			for (int c1 = n; cnt1 <= c1; c1--) {//两个背包不能写在一个for循环中
				dp[c0][c1] = max(dp[c0 - cnt0][c1 - cnt1] + 1, dp[c0][c1]);
			}
		}
	}
	return dp[m][n];
}
```
### [2809. 使数组和小于等于 x 的最少时间](https://leetcode.cn/problems/minimum-time-to-make-array-sum-at-most-x/) 不等式+贪心+DP
```c++
int minimumTime(vector<int>& nums1, vector<int>& nums2, int x) {
	//总的时间为t 对于下标i 不做任何操作到结束:nums1[i]+ t*nums2[i]
	//在第k秒做操作: (t-k)*nums2[i] 
	//k秒做操作使得数组元素和减少了nums1[i] + k*nums2[i]
	//k秒选k个元素去减少 时间是从1到k递增的 由排序不等式nums2[i]也要递增
	int n = nums1.size();
	vector<int>v(n), dp(n + 1);
	iota(begin(v), end(v), 0);
	sort(begin(v), end(v),//核心 基于排序不等式
		[&](const int& a, const int& b) {
			return nums2[a] < nums2[b]; });
	for (int i = 0; i < n; i++) {//dp[i+1][j]从0-i中选j个下标使减少量最大 最终的dp[n+1][j]是全局的最优
		int a = nums1[v[i]], b = nums2[v[i]];
		for (int j = i + 1; j; j--)//j同时也是时间k
			//dp[i + 1][j] = max(dp[i][j], dp[i][j−1] + nums1[i] + nums2[i]⋅j)
			dp[j] = max(dp[j], dp[j - 1] + a + b * j);
	}
	int s1 = accumulate(nums1.begin(), nums1.end(), 0);
	int s2 = accumulate(nums2.begin(), nums2.end(), 0);
	for (int t = 0; t <= n; t++) //s1 + s2⋅t−dp[n][t]≤*x
		if (s1 + s2 * t - dp[t] <= x)return t;
	return -1;
}
```
## *完全背包

```
完全背包：有n个物品  第i个物品的体积为w[i]  价值为v[i]
每个物品无限次重复选  求体积和不超过capacity时的最大价值和
dfs(i,c)=max(dfs(i-1,c),dfs(i,c-w[i])+v[i])
```

### [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)
```c++
int coinChange(vector<int>& coins, int amount) {
	vector<int>dp(amount + 1, INT_MAX);//dp[c]表示凑成c元所需的最小硬币数
	dp[0] = 0;//达到0金额所需的最少硬币为0
	for (int c = 1; c <= amount; c++) {//枚举余额
		for (int i = 0; i < coins.size(); i++) {//枚举硬币
			if (coins[i] <= c && dp[c - coins[i]] != INT_MAX)//可以使用该硬币coins[i]
				dp[c] = min(dp[c], dp[c - coins[i]] + 1);
		}
	}
	return (dp[amount] == INT_MAX) ? -1 : dp[amount];
}
```
[279. 完全平方数](https://leetcode.cn/problems/perfect-squares/)、
[377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)、
### [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)
```c++
int change(int amount, vector<int>& coins) {
	vector<int>dp(amount + 1, 0); //dp[i]表示凑成i元的硬币方案数
	dp[0] = 1;//存在背包余额为0的情况
	for (int i = 0; i < coins.size(); i++) { //枚举所使用的硬币
		for (int c = coins[i]; c <= amount; c++) //枚举使用该硬币能达到的金额
			dp[c] += dp[c - coins[i]];
	}
	return dp[amount];
}
```
## *组合型

### [1155. 掷骰子等于目标和的方法数](https://leetcode.cn/problems/number-of-dice-rolls-with-target-sum/)

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

## *子序列

### [1143. 最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/)

```c++
//记忆化搜索
int longestCommonSubsequence(string text1, string text2) {
	int l1 = text1.size(), l2 = text2.size();
	vector<vector<int>>memo(l1, vector<int>(l2, -1));
	function<int(int, int)>dfs = [&](int i, int j)->int {
		if (i < 0 || j < 0)return 0;
		if (~memo[i][j])return memo[i][j];
		if (text1[i] == text2[j])
			memo[i][j] = dfs(i - 1, j - 1) + 1;
		else
			memo[i][j] = max(dfs(i - 1, j), dfs(i, j - 1));
		return memo[i][j];
	};
	return dfs(l1 - 1, l2 - 1);
}
```

```c++
//递推
int longestCommonSubsequence(string text1, string text2) {
	int l1 = text1.size(), l2 = text2.size();
	vector<vector<int>>dp(l1 + 1, vector<int>(l2 + 1));
	for (int i = 0; i < l1; i++) {
		for (int j = 0; j < l2; j++) {
			if (text1[i] == text2[j]) 
				dp[i + 1][j + 1] = dp[i][j] + 1;		
			else 
				dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j]);
		}
	}
	return dp[l1][l2];
}
```

### [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)

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

## *编辑距离
### [72. 编辑距离](https://leetcode.cn/problems/edit-distance/)
pre [583. 两个字符串的删除操作](https://leetcode.cn/problems/delete-operation-for-two-strings/)
```c++
int minDistance(string word1, string word2) {
	int l1 = word1.size(), l2 = word2.size();
	vector<vector<int>>memo(l1, vector<int>(l2, -1));
	function<int(int, int)>dfs = [&](int i, int j)->int {
		if (i < 0) return j + 1;//word1插入word2前面剩余的字符
		else if (j < 0)return i + 1;//word1删除前面多余的字符
		if (~memo[i][j])return memo[i][j];
		if (word1[i] == word2[j]) //不用修改
			memo[i][j] = dfs(i - 1, j - 1);
		else //删除word1中字符、插入word2中字符、替换word1[i]为word2[j] 中的最小值
			memo[i][j] = min(min(dfs(i - 1, j), dfs(i, j - 1)), dfs(i - 1, j - 1)) + 1;
		return memo[i][j];
	};
	return dfs(l1 - 1, l2 - 1);
}
```
### [115. 不同的子序列](https://leetcode.cn/problems/distinct-subsequences/)
```c++
int numDistinct(string s, string t) {//dp[i][j] 从s前i-1个字符 变成以t[j-1]结尾字符串的个数
	int mod = 1e9 + 7, ans = 0, lens = s.size(), lent = t.size();
	vector<vector<long long>>dp(lens + 1, vector<long long>(lent + 1, 0));
	dp[0][0] = 1;
	for (int i = 1; i <= lens; i++) {
		dp[i][0] = 1;//s删除所有的字符可以变成空字符串
		for (int j = 1; j <= lent; j++) {
			if (s[i - 1] == t[j - 1]) //s[i-1]对齐t[j-1]+s[i-1]对齐t[j]
				dp[i][j] = (dp[i - 1][j - 1] + dp[i - 1][j]) % mod;
			else dp[i][j] = dp[i - 1][j];
		}
	}
	return dp[lens][lent];
}
```

## *子数组、子串
思考子数组、子串统计类问题的通用技巧:

将所有子串按照其末尾字符的下标分组

考虑两组相邻的子串：以 s[ i−1 ] 结尾的子串、以 s[ i ] 结尾的子串

以 s[ i ] 结尾的子串，可以看成是以 s[ i−1 ] 结尾的子串，在末尾添加上 s[ i ] 组成

pre简单题[53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)
### [152. 乘积最大子数组](https://leetcode.cn/problems/maximum-product-subarray/)
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
### [918. 环形子数组的最大和](https://leetcode.cn/problems/maximum-sum-circular-subarray/)
```c++
//求出最大子数组和maxS以及最小子数组和minS还有数组和S
//如果所求子数组在数组中间那么就是maxS
//如果子数组是由数组两端组成则为S-minS(S!=minS和最小的子数组不能是该数组本身)
int maxSubarraySumCircular(vector<int>& nums) {
	long long dp_max = 0, dp_min = 0;
	long long sum = 0, max_s = INT_MIN, min_s = 0;//最大子数组和max_s初始化 不能为空 min_s可以初始化为0
	for (const int& num : nums) {
		sum += num;
		dp_max = max(dp_max, (long long)0) + num;
		max_s = max(max_s, dp_max);
		dp_min = min(dp_min, (long long)0) + num;
		min_s = min(min_s, dp_min);
	}
	return (sum == min_s) ? max_s : max(max_s, sum - min_s);
}
```
### [647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/)
```c++
int countSubstrings(string s) {
	int n = s.size(), ans = 0;
	vector<vector<bool>>dp(n, vector<bool>(n, false));//dp[i][j]表示从下标i到下标j的子串是否为回文子串
	for (int i = n - 1; ~i; i--) {//枚举起点
		for (int j = i; j < n; j++) {//枚举终点
			if (s[i] == s[j]&&(j - i <= 1 || dp[i + 1][j - 1])) {//ij相邻或相同或两者中间是回文串
				ans++;
				dp[i][j] = true;	
			}
		}
	}
	return ans;
}
```
类似题目[5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)

### [718. 最长重复子数组](https://leetcode.cn/problems/maximum-length-of-repeated-subarray/)
```c++
int findLength(vector<int>& nums1, vector<int>& nums2) {
	//dp[i][j] 表示以 nums1[i-1] 和 nums2[j-1] 结尾的最长公共子数组的长度
	vector<vector<int>> dp(nums1.size() + 1, vector<int>(nums2.size() + 1, 0));
	int ans = 0;
	for (int i = 1; i < nums1.size() + 1; i++) {
		for (int j = 1; j < nums2.size() + 1; j++) {
			if (nums1[i - 1] == nums2[j - 1]) 
				dp[i][j] = dp[i - 1][j - 1] + 1;
			ans = max(ans, dp[i][j]);
		}
	}
	return ans;
}
```
## *状态机DP
### [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/) (不限交易次数)

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

### [188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)  (至多交易k次)
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
## *线性DP
### [2312. 卖木头块](https://leetcode.cn/problems/selling-pieces-of-wood/)
```c++
long long sellingWood(int m, int n, vector<vector<int>>& prices) {
	vector<vector<int>>pr(m + 1, vector<int>(n + 1));
	for (const auto& p : prices) 
		pr[p[0]][p[1]] = p[2];
	//dp[i][j]高为i宽为j的矩形木块可以得到的最多钱
	vector<vector<long long>>dp(m + 1, vector<long long>(n + 1));
	for (int i = 1; i <= m; ++i) {//枚举木板高度
		for (int j = 1; j <= n; ++j) {//枚举木板宽度
			dp[i][j] = pr[i][j];
			for (int k = 1; k <= j / 2; ++k)//枚举竖切位置 改变宽度
				dp[i][j] = max(dp[i][j], dp[i][k] + dp[i][j - k]);
			for (int k = 1; k <= i / 2; ++k)//枚举横切位置 改变高度
				dp[i][j] = max(dp[i][j], dp[k][j] + dp[i - k][j]);
		}
	}
	return dp[m][n];
}
```
## *区间DP
### [1690. 石子游戏 VII](https://leetcode.cn/problems/stone-game-vii/)
```c++
int stoneGameVII(vector<int>& stones) {
	int n = stones.size(), a = 0, b = 0;
	vector<int>pre(n + 1);//sum[low,high)=pre[high]-pre[low]
	partial_sum(stones.begin(), stones.end(), pre.begin() + 1);
	vector<vector<int>>memo(n, vector<int>(n));
	//dp(i,j)表示在区间[i,j]中选取最左边或最右边使得这次得分与下个人的得分之间的差值最大
	function<int(int, int)>dp = [&](int low, int high)->int {
		if (low == high)return 0;
		if (memo[low][high])return memo[low][high];
		memo[low][high] = max(pre[high + 1] - pre[low + 1] - dp(low + 1, high),//选择最左边sum[low+1,high+1)
			pre[high] - pre[low] - dp(low, high - 1));//选择最右边sum[low,high)
		return memo[low][high];
	};
	return dp(0, n - 1);
}
```
### [516. 最长回文子序列](https://leetcode.cn/problems/longest-palindromic-subsequence/)

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
### [1039. 多边形三角剖分的最低得分](https://leetcode.cn/problems/minimum-score-triangulation-of-polygon/) 

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
### [32. 最长有效括号](https://leetcode.cn/problems/longest-valid-parentheses/)
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
## *换根DP
### [834. 树中距离之和](https://leetcode.cn/problems/sum-of-distances-in-tree/)
```c++
vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
	vector<vector<int>> g(n); // g[x] 表示 x 的所有邻居
	for (const auto& e : edges) {
		g[e[0]].emplace_back(e[1]);
		g[e[1]].emplace_back(e[0]);
	}
	vector<int> ans(n), size(n, 1); // size[i]表示以i为根节点的子树大小 初始化为1
	function<void(int, int, int)> dfs = [&](int x, int fa, int depth) {
		ans[0] += depth; // depth 为 0 到 x 的距离
		for (const int& y : g[x]) { // 遍历 x 的邻居 y
			if (y != fa) { // 避免访问父节点
				dfs(y, x, depth + 1); // x 是 y 的父节点
				size[x] += size[y]; // 累加 x 的儿子 y 的子树大小
			}
		}
	}; 
	dfs(0, -1, 0); // 0 没有父节点 先把以0作为根节点的情况算出来
	function<void(int, int)> reroot = [&](int x, int fa) {//换根
		for (const int& y : g[x]) { // 遍历 x 的邻居 y
			if (y != fa) { // 避免访问父节点
				ans[y] = ans[x] + n - 2 * size[y];//ans[x] - size[y] + (n - size[y])
				reroot(y, x); // x 是 y 的父节点
			}
		}
	};
	reroot(0, -1); // 0 没有父节点
	return ans;
}
```
### [2581. 统计可能的树根数目](https://leetcode.cn/problems/count-number-of-possible-root-nodes/)
```c++
int rootCount(vector<vector<int>>& edges, vector<vector<int>>& guesses, int k) {
	vector<vector<int>>g(edges.size() + 1);
	for (const auto& e : edges) {
		g[e[0]].emplace_back(e[1]);
		g[e[1]].emplace_back(e[0]);
	}
	unordered_set<long>us;//想象成unordered_set<pair<int, int>>
	for (const auto& e : guesses) 
		us.emplace((long long)e[0] << 32 | e[1]);
	int ans = 0, cnt0 = 0;
	function<void(int, int)>dfs = [&](int x, int fa) {
		for (const int& y : g[x]) {
			if (y != fa) {
				cnt0 += us.count((long long)x << 32 | y);
				dfs(y, x);
			}
		}
	};
	dfs(0, -1);//求以0为根的时候的猜对个数cnt0
	function<void(int, int, int)>reroot = [&](int x, int fa, int cnt) {//换根
		ans += cnt >= k;//cnt为以x为根时猜对的次数
		for (const int& y : g[x]) {
			if (y != fa)
				reroot(y, x, cnt
					- us.count((long long)x << 32 | y)//减去原来是对的 现在改成y是根节点
					+ us.count((long long)y << 32 | x));//加上原来是错的 现在改成对的
		}
	};
	reroot(0, -1, cnt0);
	return ans;
}
```
## *树形DP

二叉树 边权型
### [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)
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
### [124. 二叉树中的最大路径和](https://leetcode.cn/problems/binary-tree-maximum-path-sum/)
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
### [2246. 相邻字符不同的最长路径](https://leetcode.cn/problems/longest-path-with-different-adjacent-characters/)
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
### [337. 打家劫舍 III](https://leetcode.cn/problems/house-robber-iii/)
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
### [968. 监控二叉树](https://leetcode.cn/problems/binary-tree-cameras/)
```c++
int minCameraCover(TreeNode* root) {
	//A:选 ：在该节点安装摄像头 俩孩子都可以是A、B、C
	//B:不选 ：在该节点的父节点安装摄像头  俩孩子不能是B
	//C:不选 ：在该节点的左/右孩子安装摄像头  俩孩子不能是B 孩子至少一个A
	function<vector<int>(TreeNode*)>dfs = [&](TreeNode* root)->vector<int> {
		if (!root)return { INT_MAX / 2,0,0 };//空结点不能安装摄像头A为无穷大 空结点不需要监控BC返回0
		vector<int>left = dfs(root->left);
		vector<int>right = dfs(root->right);
		int A = min(left[0], min(left[1], left[2])) + min(right[0], min(right[1], right[2])) + 1;
		int B = min(left[0], left[2]) + min(right[0], right[2]);
		int C = min(left[0] + right[0], min(left[0] + right[2], left[2] + right[0]));
		return { A,B,C };
	};
	vector<int>ans = dfs(root);
	return min(ans[0], ans[2]);//根结点没有父节点
}
```

## *数位DP
基础模板 不考虑前导0
```c++
string high = to_string(finish), low = to_string(start);//将数字转换成字符串表示
int n = high.size();//获得最大的位数
low = string(n - low.size(), '0') + low;//下界补上前导0 与high对齐
vector<int> memo(n, -1);//记忆化搜索
//dfs(i, limit_low, limit_high) i表示当前位 limit_为真表示前i-1位和最大/小数前缀相同
function<int(int, bool, bool)>dfs = [&](int i, bool limit_low, bool limit_high)->int {//第i个位可以枚举的范围 [low_, high_]
	if (i == n)return 1;
	if (!limit_low && !limit_high && memo[i] != -1) return memo[i];
	int res = 0;
	int low_ = limit_low ? (int)(low[i] - '0') : 0;
	int high_ = limit_high ? (int)(high[i] - '0') : 9;
	for (int d = low_; d <= high_; d++)
		res += dfs(i + 1, limit_low && d == low_, limit_high && d == high_);
	if (!limit_low && !limit_high)memo[i] = res;
	return res;
};
dfs(0, true, true);
```
### [2999. 统计强大整数的数目](https://leetcode.cn/problems/count-the-number-of-powerful-integers/)
```c++
long long numberOfPowerfulInt(long long start, long long finish, int limit, string s) {
	string high = to_string(finish), low = to_string(start);//将数字转换成字符串表示
	int n = high.size(), dif = n - s.size();
	low = string(n - low.size(), '0') + low;//下界补上前导0 与high对齐
	vector<long long> memo(n, -1);
	function<long long(int, bool, bool)>dfs = [&](int i, bool limit_low, bool limit_high)->long long {
		if (i == n)return 1;
		if (!limit_low && !limit_high && memo[i] != -1) return memo[i];
		long long res = 0;
		int low_ = limit_low ? (int)(low[i] - '0') : 0;
		int high_ = limit_high ? (int)(high[i] - '0') : 9;
		if (i < dif) {//正常模板 约束加在for循环中
			for (int d = low_; d <= min(limit, high_); d++)
				res += dfs(i + 1, limit_low && d == low_, limit_high && d == high_);
		}
		else {//只能填s[i-dif]
			int x = s[i - dif] - '0';
			if (low_ <= x && x <= min(high_, limit))
				res += dfs(i + 1, limit_low && x == low_, limit_high && x == high_);
		}
		if (!limit_low && !limit_high)memo[i] = res;
		return res;
	};
	return dfs(0, true, true);
}
//扩展模板 考虑前导0
long long numberOfPowerfulInt(long long start, long long finish, int limit, string s) {
	string high = to_string(finish), low = to_string(start);//将数字转换成字符串表示
	int n = high.size(), dif = n - s.size();
	low = string(n - low.size(), '0') + low;//下界补上前导0 与high对齐
	vector<long long> memo(n, -1);
	function<long long(int, bool, bool, bool)>dfs = [&](int i, bool limit_low, bool limit_high, bool is_num)->long long {
		if (i == n)return 1;
		if (!limit_low && !limit_high && memo[i] != -1) return memo[i];
		long long res = 0;
		if (!is_num && low[i] == '0') {//is_num 表示前面是否填了非零数 前面都是0 limit_low一定为true
			if (i < dif)
				res = dfs(i + 1, true, false, false);//这一位也可以填0
		}
		int low_ = limit_low ? (int)(low[i] - '0') : 0;
		int high_ = limit_high ? (int)(high[i] - '0') : 9;
		int d0 = is_num ? 0 : 1;
		if (i < dif) {//正常模板 约束加在for循环中
			for (int d = max(d0, low_); d <= min(limit, high_); d++)
				res += dfs(i + 1, limit_low && d == low_, limit_high && d == high_, true);
		}
		else {//只能填s[i-dif]
			int x = s[i - dif] - '0';
			if (max(d0, low_) <= x && x <= min(high_, limit))
				res += dfs(i + 1, limit_low && x == low_, limit_high && x == high_, true);
		}
		if (!limit_low && !limit_high)memo[i] = res;
		return res;
	};
	return dfs(0, true, true, false);
}
```
### [2719. 统计整数数目](https://leetcode.cn/problems/count-of-integers/)
```c++
int count(string num1, string num2, int min_sum, int max_sum) {
	long mod = 1e9 + 7; int n = num2.size();
	num1 = string(n - num1.size(), '0') + num1;
	vector<vector<int>>memo(n, vector<int>(min(9 * n, max_sum) + 1, -1));
	//截至到第i位 位和为sum 
	function<int(int, int, bool, bool)>dfs =
		[&](int i, int sum, bool limit_low, bool limit_high)->int {
		if (sum > max_sum)return 0;
		if (i == n)return (sum >= min_sum);
		if (!limit_low && !limit_high && memo[i][sum] != -1) return memo[i][sum];
		int res = 0;

		int low_ = limit_low ? (int)(num1[i] - '0') : 0;
		int high_ = limit_high ? (int)(num2[i] - '0') : 9;

		for (int d = low_; d <= high_; d++)
			res = (long long)(res + dfs(i + 1, sum + d, limit_low && d == low_, limit_high && d == high_)) % mod;
		if (!limit_low && !limit_high)memo[i][sum] = res;
		return res;
	};
	return dfs(0, 0, true, true);
}
```
### [902. 最大为 N 的数字组合](https://leetcode.cn/problems/numbers-at-most-n-given-digit-set/)
```c++
int atMostNGivenDigitSet(vector<string>& digits, int n) {
	string high = to_string(n);
	int len = high.size();
	string low = string(len - 1, '0') + digits[0];
	vector<int> memo(len, -1);//记忆化搜索
	function<int(int, bool, bool)>dfs = [&](int i, bool limit_high, bool is_num)->int {
		if (i == len)return is_num;
		if (!limit_high && is_num && ~memo[i]) return memo[i];
		int res = 0;
		if (!is_num)
			res = dfs(i + 1, false, false);
		char high_ = limit_high ? high[i] : digits[digits.size() - 1][0];
		for (const auto& d : digits) {// 枚举要填入的数字 d
			if (d[0] > high_) break;// d 超过上限
			res += dfs(i + 1, limit_high && d[0] == high_, true);
		}
		if (is_num && !limit_high)memo[i] = res;
		return res;
	};
	return dfs(0, true, false);
}
```
# 贪心
## [1029. 两地调度](https://leetcode.cn/problems/two-city-scheduling/)
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
## [134. 加油站](https://leetcode.cn/problems/gas-station/)
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
## [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

```c++
int maxProfit(vector<int>& prices) {
	int ans = 0;
	for (int i = 0; i < prices.size() - 1; i++) {
		ans += max(0, prices[i + 1] - prices[i]);
	}
	return ans;
}
```

## [1402. 做菜顺序](https://leetcode.cn/problems/reducing-dishes/)

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
## [334. 递增的三元子序列](https://leetcode.cn/problems/increasing-triplet-subsequence/)
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
## [435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/)
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
## [2952. 需要添加的硬币的最小数量](https://leetcode.cn/problems/minimum-number-of-coins-to-be-added/)
```c++
int minimumAddedCoins(vector<int>& coins, int target) {
	ranges::sort(coins);
	int ans = 0, i = 0, s = 0;//[0,s]是能够筹够的面值
	while (s < target) {
		if (i < coins.size() && coins[i] <= s + 1)
			s += coins[i++];
		else {//需要添加硬币面值为s+1
			++ans;
			s += s + 1;
		}
	}
	return ans;
}
```
# 图
遍历上下左右 四个方向
```c++
vector<int> dirs = {0, -1, 0, 1, 0};
//queue<pair<int, int>>q;
vector<pair<int, int>>cur, next;
//...
for (int d = 0; d < 4; ++d) {//将上、下、左、右坐标加入
	int nx = x + dirs[d];
	int ny = y + dirs[d + 1];
	//q.emplace(nx, ny);
	next.emplace_back(nx, ny);
}
cur = move(next);
```
遍历加上对角线八个方向
```c++
vector<int> dirs = { -1, -1, 0, -1, 1, 1, 0, 1, -1 };
//...
for (int d = 0; d < 8; d++) {
	int nx = x + dirs[d];
	int ny = y + dirs[d + 1];
	//...
}
```
## *深度优先搜索DFS
pre[200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)、
[2684. 矩阵中移动的最大次数](https://leetcode.cn/problems/maximum-number-of-moves-in-a-grid/)
### [695. 岛屿的最大面积](https://leetcode.cn/problems/max-area-of-island/)
do [1020. 飞地的数量](https://leetcode.cn/problems/number-of-enclaves/)
```c++
int maxAreaOfIsland(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size(), ans = 0;
	vector<int> dirs = { 0, -1, 0, 1, 0 };
	function<int(int, int)>dfs = [&](int x, int y) {
		if (!grid[x][y])return 0;
		grid[x][y] = 0;//标记为已访问
		int res = 1;
		for (int d = 0; d < 4; d++) {
			int nx = x + dirs[d];
			int ny = y + dirs[d + 1];
			if (0 <= nx && nx < m && 0 <= ny && ny < n)
				res += dfs(nx, ny);
		}
		return res;
	};
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (grid[i][j]) 
				ans = max(ans, dfs(i, j));
		}
	}
	return ans;
}
```
### [1466. 重新规划路线](https://leetcode.cn/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/)
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
pre[130. 被围绕的区域](https://leetcode.cn/problems/surrounded-regions/)
### [417. 太平洋大西洋水流问题](https://leetcode.cn/problems/pacific-atlantic-water-flow/)
```c++
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
	vector<int> dirs = { 0, -1, 0, 1, 0 };
	int m = heights.size(), n = heights[0].size();
	vector<vector<int>>p(m, vector<int>(n)), a(m, vector<int>(n)), ans;
	function<void(int, int, bool)>dfs = [&](int x, int y, bool isp) {
		if ((p[x][y] && isp) || (a[x][y] & (!isp))) return;
		(isp ? p[x][y] : a[x][y]) = 1;
		for (int d = 0; d < 4; ++d) {
			int nx = x + dirs[d];
			int ny = y + dirs[d + 1];
			if (0 <= nx && nx < m && 0 <= ny && ny < n && heights[nx][ny] >= heights[x][y])
				dfs(nx, ny, isp);
		}
	};
	for (int i = 0; i < m; ++i) {
		dfs(i, 0, true);//能访问Pacific Ocean
		dfs(i, n - 1, false);//能访问 Atlantic Ocean
	}
	for (int j = 0; j < n; ++j) {
		dfs(0, j, true);//能访问Pacific Ocean
		dfs(m - 1, j, false);//能访问 Atlantic Ocean
	}
	for (int x = 0; x < m; ++x) {
		for (int y = 0; y < n; ++y) {
			if (p[x][y] && a[x][y])
				ans.push_back({ x,y });
		}
	}
	return ans;
}
```
### [827. 最大人工岛](https://leetcode.cn/problems/making-a-large-island/)
```c++
int largestIsland(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size();
	vector<int> dirs = { 0, -1, 0, 1, 0 };
	vector<int>areas(2, 0); int cnt, index = 2;
	function<void(int, int, int)>dfs = [&](int x, int y, int index) {
		grid[x][y] = index;//已访问的岛屿标记为岛屿编号
		areas.back() = ++cnt;//更新岛屿的大小
		for (int d = 0; d < 4; ++d) {
			int nx = x + dirs[d];
			int ny = y + dirs[d + 1];
			if (0 <= nx && nx < m && 0 <= ny && ny < n && grid[nx][ny] == 1)
				dfs(nx, ny, index);
		}
	};
	for (int x = 0; x < m; ++x) {
		for (int y = 0; y < n; ++y) {
			if (grid[x][y] == 1) {
				areas.push_back(index);
				cnt = 0;//岛屿大小初始化为0
				dfs(x, y, index++);
			}
		}
	}
	if (areas.size() == 2)return 1;//没有岛屿
	int ans = areas[2];//初始化为第一个岛屿面积
	for (int x = 0; x < m; ++x) {
		for (int y = 0; y < n; ++y) {
			if (!grid[x][y]) {
				int res = 0;
				unordered_set<int>us;
				for (int d = 0; d < 4; ++d) {
					int nx = x + dirs[d];
					int ny = y + dirs[d + 1];
					if (0 <= nx && nx < m && 0 <= ny && ny < n)
						us.emplace(grid[nx][ny]);
				}
				for (const int& v : us)res += areas[v];
				ans = max(ans, res + 1);
			}
		}
	}
	return ans;
}
```
pre [3067. 在带权树网络中统计可连接服务器对数目](https://leetcode.cn/problems/count-pairs-of-connectable-servers-in-a-weighted-tree-network/)
### [2867. 统计树中的合法路径数目](https://leetcode.cn/problems/count-valid-paths-in-a-tree/)
```c++
long long countPaths(int n, vector<vector<int>>& edges) {
	vector<bool> cnt(n + 1, false);//false表示质数 true表示非质数
	int init = [&]() {//找到1到n中所有的质数
		cnt[1] = true;
		for (int i = 2; i * i <= n; i++) {
			if (!cnt[i]) {
				for (int j = i * i; j <= n; j += i) {
					cnt[j] = true;
				}
			}
		}
		return 0;
	}();
	vector<vector<int>>next(n + 1);//结点的领域结点
	for (const auto& e : edges) {
		next[e[0]].emplace_back(e[1]);
		next[e[1]].emplace_back(e[0]);
	}
	vector<int>size(n + 1), nodes;
	function<void(int, int)>dfs = [&](int x, int fa) {
		nodes.emplace_back(x);
		for (const int& y : next[x]) //dfs非质数结点
			if (y != fa && cnt[y]) dfs(y, x);
	};
	long long ans = 0; int sum;
	for (int x = 1; x <= n; x++) {//枚举所有质数结点
		if (cnt[x])continue;
		sum = 0;
		for (const int& y : next[x]) {//质数x视为根结点 将树分成多个连通块
			if (!cnt[y]) continue;//只能有一个质数 遍历非质数结点
			if (!size[y]) {//该连通块没有计算过
				nodes.clear();
				dfs(y, -1);//遍历y所在的连通块不经过质数的情况下有多少非质数
				for (const int& z : nodes)
					size[z] = nodes.size();
			}
			ans += (long long)size[y] * sum;//以质数x为拐点两颗子树构成的路径
			sum += size[y];//更新左侧连通块中非质数的数量
		}
		ans += sum;//仅从质数x出发到达非质数一条路径
	}
	return ans;
}
```
## *广度优先搜索BFS

### [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)
```c++
int orangesRotting(vector<vector<int>>& grid) {
	int m = grid.size(), n = grid[0].size();
	int ans = -1, res = 0, x, y;
	vector<int> dirs = { 0, -1, 0, 1, 0 };
	vector<pair<int, int>>cur, next;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (grid[i][j] == 1)res++;
			else if (grid[i][j] == 2) {
				cur.emplace_back(i, j);
			}
		}
	}
	if (!res)return 0;
	while (!cur.empty()) {
		ans++;
		for (const auto& it : cur) {
			for (int i = 0; i < 4; i++) {//遍历上下左右四个坐标
				x = it.first + dirs[i];
				y = it.second + dirs[i + 1];
				if (x < 0 || x >= m || y < 0 || y >= n)continue;
				if (grid[x][y] == 1) {
					res--;
					grid[x][y] = 2;
					next.emplace_back(x, y);
				}
			}
		}
		cur = move(next);
	}
	return res ? -1 : ans;
}
```
### [1926. 迷宫中离入口最近的出口](https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/)
```c++
int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
	int m = maze.size(), n = maze[0].size(), path = -1;
	vector<pair<int, int>>q, next;
	vector<vector<bool>>visited(m, vector<bool>(n, false));
	vector<int> dirs = { 0, -1, 0, 1, 0 };
	q.emplace_back(entrance[0], entrance[1]);
	while (!q.empty()) {
		path++;
		for (const auto& it : q) {
			int x = it.first;
			int y = it.second;
			if (x == m - 1 || x == 0 || y == 0 || y == n - 1) {//可能是出口
				if (x != entrance[0] || y != entrance[1]) {//出口
					return path;
				}
			}
			for (int d = 0; d < 4; ++d) {
				int nx = x + dirs[d];
				int ny = y + dirs[d + 1];
				if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == '.' && !visited[nx][ny]) {
					next.emplace_back(nx, ny);
					visited[nx][ny] = true;
				}
			}
		}
		q = move(next);
	}
	return -1;
}
```
### [1162. 地图分析](https://leetcode.cn/problems/as-far-from-land-as-possible/) 多源BFS
```c++
int maxDistance(vector<vector<int>>& grid) {
	int n = grid.size(), ans = -1, sum = 0;
	vector<int> dirs = { 0, -1, 0, 1, 0 };
	vector<pair<int, int>>q, next;
	vector<vector<bool>>vis(n, vector<bool>(n, false));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			sum += grid[i][j];
			if (grid[i][j]) q.emplace_back(i, j);
		}
	}
	if (!sum || sum == n * n)return -1;//全为海洋或陆地
	while (!q.empty()) {
		ans++;//距离+1
		for (const auto& it : q) {
			for (int d = 0; d < 4; ++d) {
				int nx = it.first + dirs[d];
				int ny = it.second + dirs[d + 1];
				if (0 <= nx && nx < n && 0 <= ny && ny < n && !vis[nx][ny]) {
					vis[nx][ny] = true;
					next.emplace_back(nx, ny);
				}
			}
		}
		q = move(next);
	}
	return ans;
}
```
### [127. 单词接龙](https://leetcode.cn/problems/word-ladder/)
```c++
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
	unordered_set<string>dict; unordered_map<string, int>visit;
	for (const string& s : wordList)dict.emplace(s);
	if (!dict.count(endWord))return 0;//特判 字典中不存在endWord直接返回0
	unordered_set<string> q, next; q.emplace(beginWord);//BFS 用unordered_set替换vector加速
	int ans = 0;
	while (!q.empty()) {
		ans++;
		for (string s : q) {
			if (s == endWord)return ans;
			visit[s]++;//标记遍历过的字符串
			for (int i = 0; i < s.size(); ++i) {//逐字符改变字符串
				char t = s[i];
				for (int j = 'a'; j <= 'z'; ++j) {
					s[i] = j;
					if (dict.count(s) && (!visit[s]))
						next.emplace(s);
				}
				s[i] = t;//恢复该字符 以便改变下一个字符
			}
		}
		q = move(next);
	}
	return 0;
}
```
## *拓扑排序
[1557. 可以到达所有点的最少点数目](https://leetcode.cn/problems/minimum-number-of-vertices-to-reach-all-nodes/)
### [207. 课程表](https://leetcode.cn/problems/course-schedule/) 有向图
```c++
bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
	vector<int>indegree(numCourses);//记录入度
	vector<vector<int>>next(numCourses);
	for (const auto& it : prerequisites) {
		++indegree[it[0]];
		next[it[1]].push_back(it[0]);
	}
	queue<int>q;//记录入度为0的结点
	for (int i = 0; i < numCourses; ++i)
		if (indegree[i] == 0)
			q.push(i);
	int cnt = 0;//记录已经学过的课程数量
	while (!q.empty()) {
		int t = q.front();
		q.pop();
		++cnt;
		for (const auto& ne : next[t])
			if (--indegree[ne] == 0)
				q.push(ne);
	}
	return cnt == numCourses;
}
```
### [310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/) 无向图
```c++
vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
	if (n == 1)return { 0 };
	vector<int>depth(n), ans;
	vector<vector<int>> g(n);
	for (const auto& e : edges) {
		g[e[0]].emplace_back(e[1]);
		g[e[1]].emplace_back(e[0]);
		++depth[e[0]];
		++depth[e[1]];
	}
	queue<int>qu;//存放度为1的结点
	for (int i = 0; i < n; ++i) 
		if (depth[i] == 1)qu.emplace(i);
	
	int res = n;//剩余结点
	while (res > 2) {//不断删除度为1的结点 直到结点数小于等于2
		int qz = qu.size();
		res -= qz;
		for (int i = 0; i < qz; ++i) {
			int cur = qu.front();
			qu.pop();
			for (const auto& v : g[cur]) 
				if (--depth[v] == 1)qu.emplace(v);
		}
	}
	while (!qu.empty()) {
		ans.emplace_back(qu.front());
		qu.pop();
	}
	return ans;
}
```
## 树上倍增、最近公共祖先LCA
### [1483. 树节点的第 K 个祖先](https://leetcode.cn/problems/kth-ancestor-of-a-tree-node/)
```c++
class TreeAncestor {
private:
	vector<vector<int>>pa;
public:
	TreeAncestor(int n, vector<int>& parent) {
		int m = 32 - __builtin_clz(n); // n 的二进制长度
		pa.resize(n, vector<int>(m, -1));
		for (int i = 0; i < n; ++i)
			pa[i][0] = parent[i];
		for (int i = 0; i < m - 1; ++i)
			for (int x = 0; x < n; ++x)
				if (int p = pa[x][i]; p != -1)
					pa[x][i + 1] = pa[p][i];
	}

	int getKthAncestor(int node, int k) {
		int m = 32 - __builtin_clz(k);
		for (int i = 0; i < m; ++i)
			if ((k >> i) & 1) { // k 的二进制从低到高第 i 位是 1
				node = pa[node][i];
				if (node < 0)break;
			}
		return node;
	}
};
```
### [2846. 边权重均等查询](https://leetcode.cn/problems/minimum-edge-weight-equilibrium-queries-in-a-tree/)
```c++
vector<int> minOperationsQueries(int n, vector<vector<int>>& edges, vector<vector<int>>& queries) {
	vector<vector<pair<int, int>>> g(n);//记录节点i相连的节点id以及边权重
	for (const auto& e : edges) {
		g[e[0]].emplace_back(e[1], e[2] - 1);
		g[e[1]].emplace_back(e[0], e[2] - 1);
	}

	int m = 32 - __builtin_clz(n); // n 的二进制长度

	/*  树上倍增:
	pa[x][i] 表示x的第2^i个祖先节点
	pa[x][0] = parent[x]
	pa[x][1] = pa[pa[x][0]][0] = parent[parent[x]]
	pa[x][i+1] = pa[pa[x][i]][i]*/
	vector<vector<int>> pa(n, vector<int>(m, -1));
	vector<int> depth(n);
	//cnt[x][i][j]表示节点x到其第2^i个祖先的路径上，边权值为j的边的数量
	vector<vector<array<int, 26>>> cnt(n, vector<array<int, 26>>(m));
	function<void(int, int)> dfs = [&](int x, int fa) {
		pa[x][0] = fa;
		for (const auto &ne : g[x]) {//当前节点相连的下一个节点们
			if (ne.first != fa) {//下个节点不是当前节点
				cnt[ne.first][0][ne.second] = 1;
				depth[ne.first] = depth[x] + 1;//下个节点的深度+1
				dfs(ne.first, x);
			}
		}
	};
	dfs(0, -1);//将节点0视为树的根节点 其父节点为-1

	for (int i = 0; i < m - 1; i++) {
		for (int x = 0; x < n; x++) {
			int p = pa[x][i];
			if (p != -1) {
				int pp = pa[p][i];
				pa[x][i + 1] = pp;
				for (int j = 0; j < 26; ++j) {
					cnt[x][i + 1][j] = cnt[x][i][j] + cnt[p][i][j];
				}
			}
		}
	}
	/*保留出现次数最多的边 其余的全部修改
	从 a 到 b 的路径长度(边数)
		depth[a] + depth[b] - 2 * depth[lca]
	从 a 到 b 出现次数最多的边
	1 <= wi <= 26 统计每种边权的出现次数*/
	vector<int> ans;
	for (auto& q : queries) {
		int x = q[0], y = q[1];
		int path_len = depth[x] + depth[y]; // 最后减去 depth[lca] * 2
		vector<int>cw(26);
		if (depth[x] > depth[y]) {
			swap(x, y);//保证x深度小于y
		}
		//求lca:
		
		// 让 y 和 x 在同一深度
		for (int k = depth[y] - depth[x]; k; k &= k - 1) {
			int i = __builtin_ctz(k);//本次跳跃的深度
			int p = pa[y][i];
			for (int j = 0; j < 26; ++j) {
				cw[j] += cnt[y][i][j];//统计本次跳跃增加了多少路径权值
			}
			y = p;
		}
		//x和y一起跳
		if (y != x) {
			for (int i = m - 1; ~i; i--) {// x 和 y 同时上跳 2^i 步
				int px = pa[x][i], py = pa[y][i];
				if (px != py) {//同时往上跳i步 此时px!=py说明lca还在上面 可以跳
					for (int j = 0; j < 26; j++) {
						cw[j] += cnt[x][i][j] + cnt[y][i][j];
					}
					x = px;
					y = py; 
				}
			}
			//最后跳完之后lca就在x和y的上面 还需跳一步
			for (int j = 0; j < 26; j++) {
				cw[j] += cnt[x][0][j] + cnt[y][0][j];
			}
			x = pa[x][0];//x即为lca
		}

		path_len -= depth[x] * 2;
		ans.emplace_back(path_len - *max_element(cw.begin(), cw.end()));
	}
	return ans;
}
```
# 博弈
## [486. 预测赢家](https://leetcode.cn/problems/predict-the-winner/)
```c++
bool predictTheWinner(vector<int>& nums) {
	int n = nums.size();
	vector<vector<int>>dp(n, vector<int>(n, -1));
	vector<int>pre(n + 1);//sum[low, high] = pre[high + 1] - pre[low]
	partial_sum(nums.begin(), nums.end(), pre.begin() + 1);
	function<int(int, int)>dfs = [&](int low, int high)->int {
		if (dp[low][high] != -1) return dp[low][high];
		if (low == high) return nums[low];
		dp[low][high] = pre[high + 1] - pre[low] - min(dfs(low + 1, high), dfs(low, high - 1));
		return dp[low][high];
	};//dfs表示在区间中能拿到的最优解(留给下一个拿的最少)
	return dfs(0, n - 1) * 2 >= pre[n];
}
```
