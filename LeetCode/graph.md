### 题号：98 图中所有可达路径
#### 思路：
- 构建邻接矩阵：n个节点m条边
	- 构建一个n × n的二维数组，对于有边的两个节点矩阵中置为1
- dfs：
	- 参数：邻接矩阵，当前元素，要找的元素
	- 终止：当前元素 = n，把当前路径加入到结果中
	- 当前节点处理：遍历当前节点能到的节点，也就是矩阵中值为1的节点，从把能到的节点作为当前节点
#### c++实现：
```c++

```
#### go实现：
##### 邻接矩阵
- 创建节点个数平方的矩阵
```go
package execute

import (
	"fmt"
)

var result [][]int
var path []int

func allPathdfs(graph [][]int, x int, n int) {
	if x == n {
		tmp := make([]int, len(path))
		copy(tmp, path)
		result = append(result, tmp)
		return
	}

	for i := 1; i <= n; i++ {
		if graph[x][i] == 1 {
			path = append(path, i)
			allPathdfs(graph, i, n)
			path = path[:len(path)-1]
		}
	}
}

func Graph() {
	var m, n int
	fmt.Scanf("%d %d", &m, &n)

	graph := make([][]int, n+1)
	for i := range graph {
		graph[i] = make([]int, n+1)
	}
	fmt.Println(len(graph))

	for m > 0 {
		var s, t int
		fmt.Scanf("%d %d", &s, &t)
		graph[s][t] = 1
	}
	// from 1 to n
	path = append(path, 1)
	allPathdfs(graph, 1, n)

	if len(result) == 0 {
		fmt.Println(-1)
	} else {
		for _, pa := range result { // the last elem has no space
			for i := 0; i < len(pa)-1; i++ {
				fmt.Print(pa[i], " ")
			}
			fmt.Println(pa[len(pa)-1])
		}
	}
}

```
##### 邻接表
- 每个元素一行，有相连的才在对应位置置1
#### 总结：
- 没什么需要注意的，就是从第一个节点开始，然后遍历根当前节点有链接的节点，只到遍历到想要的节点


### 题号：99 number of island
#### 思路：
分成深搜和广搜，都写一下
##### 深搜
- 遍历所有节点，对于没被标记的陆地节点将与它相连的节点都标记上，并且将岛屿数量 + 1，这里包含两个过程
	- 一个是遍历所有节点
		- 直接两个循环，循环的目的一个是对岛屿计数，一个是标记已经遍历过的节点，所以得整一个和地图一样大的标记数组用来标记一个节点是否被访问过
	- 一个是对于每一个节点找到与它相连的节点
		- 这得递归，对于每个节点都找它的上下左右是否存在，如果存在就标记上
		- 对于找上下左右，可以用一个4 × 2的数组，4代表4个方向，2代表两个维度
##### 广搜
- 用一个队列记录当前节点周围的节点，当一个节点入队的时候就标记为访问过的，然后从队头拿出节点，拿出的节点就是某个节点周围的节点
#### c++实现：
```c++

```
#### go实现：
```go
var dir [4][2]int = [4][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}

// position of current node
func dfs(gd [][]int, vd [][]bool, x, y int) {
	for _, d := range dir {
		nextX := x + d[0]
		nextY := y + d[1]
		fmt.Println(d, nextX, nextY)
		if nextX < 0 || nextX >= len(gd) || nextY < 0 || nextY >= len(gd[0]) {
			continue
		}

		if !vd[nextX][nextY] && gd[nextX][nextY] == 1 {
			vd[nextX][nextY] = true
			dfs(gd, vd, nextX, nextY)
		}
	}
}

func main() {
	result := 0
	var m, n int
	fmt.Scanf("%d %d", &n, &m)

	grid := make([][]int, n)
	for i := 0; i < n; i++ {
		grid[i] = make([]int, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			var elem int
			fmt.Scan(&elem)
			grid[i][j] = elem
		}
	}

	fmt.Println(grid)
	visited := make([][]bool, n)
	for i, _ := range visited {
		visited[i] = make([]bool, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if grid[i][j] == 1 && !visited[i][j] {
				result++
				visited[i][j] = true
				dfs(grid, visited, i, j)
			}
		}
	}

	fmt.Println(result)
}
```
```go
package main

import(
    "fmt"
)
var dir [4][2]int = [4][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}

type pair struct {
	x int
	y int
}

var que []pair

func bfs(gd [][]int, vd [][]bool, x, y int) {
	que = append(que, pair{x, y})
	vd[x][y] = true
	for len(que) > 0 {
		curx := que[0].x
		cury := que[0].y
		que = que[1:]
		for _, d := range dir {
			nextx := curx + d[0]
			nexty := cury + d[1]
			if nextx < 0 || nextx >= len(gd) || nexty < 0 || nexty >= len(gd[0]) {
				continue
			}
			if !vd[nextx][nexty] && gd[nextx][nexty] == 1 {
				que = append(que, pair{nextx, nexty})
				vd[nextx][nexty] = true // mark only when push into queue
			}
		}
	}
}

func main() {
	result := 0
	var m, n int
	fmt.Scanf("%d %d", &n, &m)

	grid := make([][]int, n)
	for i := 0; i < n; i++ {
		grid[i] = make([]int, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			var elem int
			fmt.Scan(&elem)
			grid[i][j] = elem
		}
	}

	fmt.Println(grid)
	visited := make([][]bool, n)
	for i, _ := range visited {
		visited[i] = make([]bool, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if grid[i][j] == 1 && !visited[i][j] {
			    fmt.Println(i,j,visited)
				result++
				bfs(grid, visited, i, j)
			}
		}
	}

	fmt.Println(result)
}


```
#### 总结：
- 看清题目，特别是行和列要分清
- 广搜的时候没有递归，深搜才有
### 题号：100 max area of island
#### 思路：
- 就是每次遇到岛屿记录最大面积
#### c++实现：
```c++

```
#### go实现：
```go
package main
import (
    "fmt"
)
var dir [4][2]int = [4][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
func dfs(gd [][]int, vd [][]bool, x, y int) {
	for _, d := range dir {
		nextX := x + d[0]
		nextY := y + d[1]
	
		if nextX < 0 || nextX >= len(gd) || nextY < 0 || nextY >= len(gd[0]) {
			continue
		}

		if !vd[nextX][nextY] && gd[nextX][nextY] == 1 {
			count++
			vd[nextX][nextY] = true
			fmt.Println(count, vd)
			dfs(gd, vd, nextX, nextY)
		}
	}
}
func max(nums... int)int {
	max := nums[0]
	for _, num :=range nums{
		if num > max{
			max = num
		}
	}
	return max
}
var count int = 0

func main() {
	result := 0
	var m, n int
	fmt.Scanf("%d %d", &n, &m)

	grid := make([][]int, n)
	for i := 0; i < n; i++ {
		grid[i] = make([]int, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			var elem int
			fmt.Scan(&elem)
			grid[i][j] = elem
		}
	}

	fmt.Println(grid)
	visited := make([][]bool, n)
	for i, _ := range visited {
		visited[i] = make([]bool, m)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if grid[i][j] == 1 && !visited[i][j] {
				count = 1
				visited[i][j] = true
				dfs(grid, visited, i, j)
				fmt.Println(count, result)
				result = max(count, result)
				fmt.Println(result)
			}
		}
	}

	fmt.Println(result)
}

```
#### 总结：
- 
### 题号：110 string concatanation
#### 思路：
- 广度优先搜索：从一个节点出发，
#### c++实现：
```c++

```
#### go实现：
```go
package main
import("fmt")
func main() {
	var n int
	fmt.Scanf("%d", &n)

	var beginStr, endStr string
	fmt.Scanf("%s %s", &beginStr, &endStr)
	fmt.Println("bs", beginStr, "es", endStr)

	strMap := make(map[string]struct{})
	for n > 0 {
		var inputStr string
		fmt.Scanln(&inputStr)
		strMap[inputStr] = struct{}{}
		n--
	}
    fmt.Println("strMap", strMap)
	que := make([]string, 0)
	visitedMap := make(map[string]int)
	que = append(que, beginStr)
	visitedMap[beginStr] = 1

	for len(que) > 0 {
		word := que[0]
		fmt.Println("word", word)
		que = que[1:]
		bWord := []rune(word)
		for posOfWord := 0; posOfWord < len(word); posOfWord++ {
			for i := 0; i < 26; i++ {
				bWord[posOfWord] = rune(i + 'a')
				fmt.Println("bWord", string(bWord))
				if string(bWord) == endStr {
				    fmt.Println("get it")
					fmt.Print(visitedMap[word] + 1)
					return
				}
				_, exists1 := strMap[string(bWord)]
				_, exists2 := visitedMap[string(bWord)]
				if exists1 && !exists2 {
					visitedMap[string(bWord)] = visitedMap[word] + 1
					que = append(que, string(bWord))
					fmt.Println("add", string(bWord))
				}
			}
			bWord = []rune(word)
		}
	}
	fmt.Println(0)
}
```
#### 总结：
- 一个要注意的点是：当使用一个变量在一个位置上完成替换后要回复到原来的变量字符串，就是上面的 bWord = [] rune(word)
### 题号：105 completely accessible
#### 思路：
- 深搜，用一个切片记录哪些节点是能访问到的，最后遍历是否所有节点都能访问
#### c++实现：
```c++

```
#### go实现：
```go
package main

import "fmt"

func dfs(graph [][]int, key int, visited []bool) {
	for i := 1; i < len(graph[0]); i++ {
	    fmt.Println("graph[key][i]",graph[key][i])
		if graph[key][i] == 1 && !visited[i] {
			visited[i] = true
			dfs(graph, i, visited)
		}
	}
}

func main() {
	var n, k int
	fmt.Scanf("%d %d", &n, &k)

	graph := make([][]int, n+1)
	for i, _ := range graph {
		graph[i] = make([]int, n+1)
	}
    
	for k > 0 {
		var i, j int
		fmt.Scanf("%d %d", &i, &j)
		graph[i][j] = 1
		k--
	}
	fmt.Println(graph)

	visited := make([]bool, n+1)
	visited[1] = true

	dfs(graph, 1, visited)

	for i := 1; i <= n; i++ {
		if visited[i] == false {
		    fmt.Println("out", i)
			fmt.Println(-1)
			return
		}
	}
	fmt.Println(1)
}

```
#### 总结：

## 并查集
- 用来判断多个元素是否在一个元素
- 也就是无向图中是否连通
### 实现
#### 初始化
- 一开始每一个元素都是一个集合，没有父子关系
#### 加入集合
- 如果两个元素的头结点不是同一个，将其中一个v加入另一个u所在的集合，方法就是将v的父节点设置为u
#### 查找头结点
- 如果当前节点就是所在集合的头结点，那么返回当前节点
- 否则找当前节点的头节点的头节点
#### 是否同一个头结点
- 判断两个节点的头结点是否是一个

### 题号：107 exists path
#### 思路：
- 就是看两个元素是否是连通的
#### c++实现：
```c++

```
#### go实现：
```go
package main

import (
	"fmt"
)

func initFather(n int) []int {
	father := make([]int, n)
	for i := 0; i < n; i++ {
		father[i] = i
	}
	return father
}

func find(father []int, u int) int {
	if father[u] == u {
		return father[u]
	}
	father[u] = find(father, father[u]) // find the father of u
	return father[u]
}

func join(father []int, u, v int) {
	rootU := find(father, u)
	rootV := find(father, v)
	if rootU != rootV {
		father[v] = u
	}
}
func isSame(father []int, u, v int) bool {
	return find(father, u) == find(father, v)
}

func main() {
	var n, m, source, target int
	fmt.Scanf("%d %d", &n, &m)

	father := initFather(n)

	for m > 0 {
		var s, t int
		fmt.Scanf("%d %d", s, t)
		join(father, s, t)
		m--
	}

	fmt.Scanf("%d %d", source, target)

	if isSame(father, source, target){
		fmt.Println(1)
	}else{
	    fmt.Println(-1)
	}
}

```
#### 总结：
- 