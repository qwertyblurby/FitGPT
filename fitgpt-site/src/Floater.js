import React, { Suspense } from 'react';

function Floater({ titleText, description, iconSrc, boxBg }) {
	const Icon = React.lazy(() => import(`./assets/icons/${iconSrc}`));
	
	return (
		<Suspense fallback={<div>Loading...</div>}>
			<div className={"flex flex-col items-center space-y-3"+(boxBg ? " bg-gray-800 rounded-lg p-6" : "")}>
				<Icon className="h-10 w-10 text-gray-50" />
				<h3 className="text-center text-lg font-semibold">{titleText}</h3>
				<p className="text-center text-gray-400">{description}</p>
			</div>
		</Suspense>
	)
}

export default Floater;
