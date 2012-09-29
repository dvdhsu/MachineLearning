function [i, j] = findMinPosition(zebra)
	[minRow, minRowPosition] = min(zebra);
	[minCol, minColPosition] = min(minRow);
	i = minRowPosition(minColPosition);
	j = minColPosition;
end

