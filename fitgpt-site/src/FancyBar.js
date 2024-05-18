import { useState, useEffect, useRef } from 'react';
import { colorCodes, colorMapping, lightColors } from './ModelData';

function FancyBar({ color, percent }) {
	const hexCode = colorCodes[color];
	const colorName = colorMapping[color];
	const labelRef = useRef(null);
	const [labelInBar, setLabelInBar] = useState(false);
	
	// Function to determine label position
	useEffect(() => {
		const labelWidth = labelRef.current.offsetWidth;
		const filledBarWidth = (percent / 100) * labelRef.current.parentElement.offsetWidth;
		if (labelWidth < filledBarWidth - 10) {
			setLabelInBar(true);
		} else {
			setLabelInBar(false);
		}
	}, [percent]);

	return (
		<div className="flex items-center justify-between">
			<div className="progress flex items-center h-6 bg-gray-700 group border-2 border-gray-700">
				<div
					className="h-5 rounded-box"
					style={{
						width: `${percent}%`,
						backgroundColor: hexCode
					}}
				/>
				<p
					ref={labelRef}
					className={`${labelInBar && "absolute left-0"} pl-3 text-left opacity-0 group-hover:opacity-100 transition-opacity`}
					style={{ color: lightColors.includes(color) && labelInBar ? "black" : "white"}}
				>
					{colorName}
				</p>
			</div>
			<p className="w-10 ml-2 text-right">{Math.round(percent)}%</p>
		</div>
	)
}

export default FancyBar;
