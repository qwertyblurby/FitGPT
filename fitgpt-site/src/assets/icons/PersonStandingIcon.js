function PersonStandingIcon(props) {
	return (
		<svg
			{...props}
			xmlns="http://www.w3.org/2000/svg"
			width="24"
			height="24"
			viewBox="0 0 24 24"
			fill="none"
			stroke="currentColor"
			strokeWidth="2"
			strokeLinecap="round"
			strokeLinejoin="round"
		>
			<circle cx="12" cy="5" r="1" />
			<path d="m9 20 3-6 3 6" />
			<path d="m6 8 6 2 6-2" />
			<path d="M12 10v4" />
		</svg>
	)
}

export default PersonStandingIcon;
