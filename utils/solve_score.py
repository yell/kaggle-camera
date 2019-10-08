N = 2640 * 0.64 # 1689.6
N = 1690
nx, ny = N/2, N/2
SCORE = 0.987976
print SCORE * nx * 10 # 8348.3972

def solve(c, nx=nx, ny=ny):
	x = c
	y = -2 * c
	while x >= 0:
		x -= 3
		y += 7
		if 0 <= x <= nx and 0 <= y <= ny:
			print x, y
	print "nothing found"

# solve(8348)
'''
845 811
842 818
839 825
836 832
833 839
'''
solve(int(0.979880*845*10)+1)
'''
843 816
840 823
837 830
834 837
831 844
'''