SELECT col1 AS c1 FROM emp Where __col1__ =/</>/<=/>=/<> value_1 AND/OR c2 IS (NOT) NULL LIMIT 5;     -> __Priority of not,and,or__

SELECT ename||'Work as a '||job as msg from emp WHERE deptno=10   #CONCAT function

#CASE
SELECT ename,sal,
	case WHEN sal <= 2000 then 'UNDERPAID'
    	  WHEN sal >= 4000 then 'OVERPAID'
          ELSE 'OK'
    END AS status
FROM emp
